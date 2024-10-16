from fastapi import FastAPI, Depends, HTTPException
from botocore.exceptions import NoCredentialsError
from database import get_db, engine, SessionLocal
from sqlalchemy.orm import Session
from schemas import ArticleSchema
from dotenv import load_dotenv
from pydantic import BaseModel
from models import Article
from openai import OpenAI
from typing import List
import requests
import models
import boto3
import json
import os
import re

models.Base.metadata.create_all(bind=engine)
Article.__table__.create(bind=engine, checkfirst=True)
db: Session = SessionLocal()

load_dotenv()
app = FastAPI()

paraphrase_api_key = os.environ.get("PARAPHRASE_KEY")
zero_gpt_api_key = os.environ.get("ZERO_GPT_API_KEY")
zero_gpt_api_url = os.environ.get("ZERO_GPT_API_URL")
aws_bucket_name = os.environ.get("AWS_BUCKET_NAME")
aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_key = os.environ.get("AWS_SECRET_KEY")
aws_region = os.environ.get("AWS_REGION")
api_key = os.environ.get("CHATBOT_KEY")

# WordPress Configuration
wp_site_url = os.getenv("WP_SITE_URL")
wp_username = os.getenv("WP_USERNAME")
wp_password = os.getenv("WP_PASSWORD")

# Define the API endpoint for creating a new post in WordPress
endpoint = f"{wp_site_url}/wp-json/wp/v2/posts"
media_endpoint = f"{wp_site_url}/wp-json/wp/v2/media"
tags_endpoint = f"{wp_site_url}/wp-json/wp/v2/tags"

# Define S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region,
)

local_file_name = "images/image_generated.png"
local_file_name2 = "images/image_generated1.png"
wp_image_name = "images/wp_image_name.png"

# Input Model for request body
class ArticleRequest(BaseModel):
    title: str

class ImageRequest(BaseModel):
    article_title: str
    article_content: str

class PostToWordPressRequest(BaseModel):
    article_id: int
    post_date: str
    wp_category_id: int
    
class GenerateAndPostRequest(BaseModel):
    title: str
    wp_category_id: int
    post_date: str
    
# Generate an article based on a title
@app.post("/generate-article-api/")
async def generate_article_api(request: ArticleRequest):
    try:
        article_data = generate_article(request.title)
        return article_data

    except HTTPException as http_exc:
        raise http_exc  # Re-raise known HTTP exceptions with the proper status code and message

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the article: {str(e)}")

def generate_article(title: str):
    try:
        prompt = (
            f"As a talented writer and SEO expert, write a detailed article on my topic '{title}'."
            "The article's length must be 2000 words and it is a must."
            "Paragraph length must be 300 words"
            "Add heading to paragraph with : where needed"
            "Maximum number of paragraphs should be 10"
            "Always give a SEO optimised title as 'Title:' that complements the article at the start of article"
            "Write like a human, with more spoken language. It is a must!"
            "For every new paragraph start put --- in the start"
            "No spelling or grammer mistake should be in any paragraph"
            "After you have written the article, you need to do the following things:"
            "Make a paragraph with heading 'SEO Suggestions'"
            "Suggest primary keyword for my content."
            "Suggest tags for my content."
            "Suggest a kickass meta description within 160 characters for my content."
        )

        client = OpenAI(api_key=api_key)
        response = generate_response(client, prompt)
        
        max_attempts = 10
        attempt_count = 0
        
        # Check the detection result to determine if the article is good to go
        while not (
            response["detection_result"]["success"]
            and response["detection_result"]["data"]["fakePercentage"] < 40
            and response["detection_result"]["data"]["textWords"] > 1000
        ):
            attempt_count += 1
            if attempt_count > max_attempts:
                raise Exception("The limit of attempts has exceeded. Please change the keyword in the title.")
            
            print("Regenerating Article as the conditions are not met...")
            response = generate_response(client, prompt)

        print("Article is good to go")

        title = response["seo_suggestions"]["title"]
        article_image1 = generate_image_from_article(
            response["seo_suggestions"]["first_half"], title
        )
        article_image2 = generate_image_from_article(
            response["seo_suggestions"]["second_half"], title
        )

        print("Images Generated Successfully")

        download_image(article_image1["image_url"], local_file_name)
        print("Image 1 downloaded")

        download_image(article_image2["image_url"], local_file_name2)
        print("Image 2 downloaded")

        cleaned_title = title.replace("<p>", "").replace("</p>", "").strip()
        final_title = cleaned_title.replace(" ", "_").lower()

        print("Final Title is: ", final_title)

        # S3 object name (path in the bucket)
        s3_file_name1 = f"articles_images/{final_title}.png"
        s3_file_name2 = f"articles_images/{final_title}2.png"

        # Upload the image to S3
        article_image_url1 = upload_image_to_s3(local_file_name, aws_bucket_name, s3_file_name1)
        article_image_url2 = upload_image_to_s3(local_file_name2, aws_bucket_name, s3_file_name2)

        article_images = [article_image_url1, article_image_url2]
        print("Images has been uploaded")
        # Return the generated article and AI detection result
        
        tags = get_tags(response["seo_suggestions"]['seo_suggestions'])
        article = create_article(response, cleaned_title, article_images, title, tags)
        print("Article has been stored in database", article.id)
        
        generate_html_file(response['seo_suggestions']['article'], cleaned_title, article_images)
        return {
            "id": article.id,
            "seo_suggestions": response["seo_suggestions"],
            "ai_detection": response["detection_result"],
            "article_image1": article_image_url1,
            "article_image2": article_image_url2,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def generate_response(client, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a talented article writer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
        )
        answer = response.choices[0].message.content
        formatted_article = format_article(answer)
        seo_suggestions = clean_article(formatted_article)
        detection_result = detect_ai_content(formatted_article)

        return {
            "detection_result": detection_result,
            "seo_suggestions": seo_suggestions,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )

def detect_ai_content(text):
    # Ensure the text is not empty or None
    if not text or text.strip() == "":
        return {"error": "Input text cannot be empty or None"}

    headers = {"ApiKey": zero_gpt_api_key, "Content-Type": "application/json"}
    payload = json.dumps(
        {
            "text": text,
            "input_text": text,
        }
    )

    try:
        response = requests.post(zero_gpt_api_url, headers=headers, data=payload)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
    except Exception as err:
        print(f"An error occurred: {err}")
        return {"error": str(err)}


def extract_main_title(title):
    match = re.search(r":\s*(.*)", title)
    if match:
        main_title = match.group(1).strip()
        return main_title
    return title


def format_article(text):
    cleaned_text = re.sub(r"[^A-Za-z0-9,.?!;:\-\(\)\s]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    formatted_text = cleaned_text.replace(" ## ", "\n\n## ")
    formatted_text = formatted_text.replace(" ### ", "\n\n### ")
    formatted_text = formatted_text.replace(" - ", "\n\n- ")
    formatted_text = formatted_text.replace(" 1. ", "\n\n1. ")

    return formatted_text


def clean_article(response):
    # Remove inline styles from the response
    cleaned_answer = re.sub(r"style=\"[^\"]*\"", "", response)

    # Insert a new paragraph tag where '---' appears
    cleaned_answer = re.sub(r"---", "</p><p>", cleaned_answer)

    title_match = re.search(r"Title:\s*(.*?)(?=<p>|---|$)", cleaned_answer, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    final_title = ""
    # Remove the title line from the cleaned_answer
    if title:
        cleaned_answer = cleaned_answer.replace(f"Title: {title}", "", 1)
        final_title = extract_main_title(title)
    seo_suggestions = ""
    seo_match = re.search(
        r"(SEO Suggestions.*?)(?=---|<\/p>|$)", cleaned_answer, re.DOTALL
    )
    if seo_match:
        seo_suggestions = seo_match.group(1).strip()
        cleaned_answer = cleaned_answer.replace(seo_suggestions, "")

    cleaned_answer = cleaned_answer.strip()

    words = cleaned_answer.split()

    # Set the maximum word limit for each half
    max_words = 350

    # Get the first half up to 350 words
    first_half_words = words[:max_words]

    # Get the second half up to 350 words (starting from where the first half ended)
    second_half_words = words[max_words : max_words * 2]

    # Join the words back into strings
    first_half = " ".join(first_half_words)
    second_half = " ".join(second_half_words)

    total_paragraphs = cleaned_answer.count("<p>") + 1

    title_pattern = re.compile(r'([A-Z][\w\s]+:)\s')

    # Split the text into paragraphs by paragraph tags or periods followed by space
    paragraphs = re.split(r'(</p>|\. )', cleaned_answer)

    formatted_paragraphs = []

    for paragraph in paragraphs:
        # Search for a title at the beginning of the paragraph
        match = title_pattern.search(paragraph)

        if match:
            title = match.group(1)
            rest_of_paragraph = paragraph[match.end():]
            title = re.sub(r":", "", title)
            
            # Add the formatted title (bold and on a separate line)
            formatted_paragraphs.append(f"<b><p>{title}</p></b>{rest_of_paragraph}")
        else:
            # No title, just append the paragraph as it is
            formatted_paragraphs.append(paragraph)

    # Join the paragraphs back into a single formatted string
    cleaned_answer = "\n\n".join(formatted_paragraphs)

    cleaned_answer = re.sub(r'\n\n\s*\.\s*\n\n', '. ', cleaned_answer)  # Fix space before period
    cleaned_answer = re.sub(r'\n\n', ' ', cleaned_answer)  # Remove unnecessary line breaks

    return {
        "article": cleaned_answer,
        "seo_suggestions": seo_suggestions,
        "total_paragraphs": total_paragraphs,
        "title": final_title,
        "first_half": first_half,
        "second_half": second_half,
    }

def generate_html_file(article_data, title, images):
    paragraphs = article_data.split("</p>")
    
    # Remove any empty paragraphs that might result from splitting.
    paragraphs = [para for para in paragraphs if para.strip()]

    # Add the closing `</p>` tag back to each paragraph to preserve HTML structure.
    paragraphs = [para + "</p>" for para in paragraphs]

    print("Length of paragraphs", len(paragraphs))
    # Insert images after specific paragraphs.
    if(len(paragraphs) == 7):
        if len(paragraphs) > 2:
            paragraphs.insert(2, f'<img width="700px" src="{images[0]}" alt="Image 1"/>')
        if len(paragraphs) > 5:
            paragraphs.insert(6, f'<img width="700px" src="{images[1]}" alt="Image 2"/>')
    else:
        if len(paragraphs) > 6:
            paragraphs.insert(6, f'<img width="700px" src="{images[1]}" alt="Image 2"/>')

    # Join the paragraphs and images back into a single HTML string.
    modified_article_content = "\n".join(paragraphs)

    # Define the complete HTML content.
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
  
    <body>
        <div class="content">
            {modified_article_content}
        </div>
    </body>
    </html>
    """

    # Write the HTML content to an HTML file.
    with open("api_response.html", "w") as html_file:
        html_file.write(html_content)

    print("HTML file created successfully: api_response.html")

    # Write the HTML content to an HTML file
    with open("api_response.html", "w") as html_file:
        html_file.write(html_content)

    print("HTML file created successfully: api_response.html")
    return html_content

def download_image(image_url: str, local_file_name: str):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(local_file_name, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Image downloaded successfully: {local_file_name}")
    else:
        raise Exception(
            f"Failed to download image. Status code: {response.status_code}"
        )


def upload_image_to_s3(local_file_name: str, bucket_name: str, s3_file_name: str):
    try:
        s3_client.upload_file(
            local_file_name,
            bucket_name,
            s3_file_name,
            ExtraArgs={
                "ContentType": "image/png",
                "ContentDisposition": "inline",
            },
        )
        print(f"Image uploaded to S3 successfully: s3://{bucket_name}/{s3_file_name}")
        
        image_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_name}"
        print(f"Image uploaded successfully. URL: {image_url}")
        return image_url
    except FileNotFoundError:
        print(f"The file was not found: {local_file_name}")
    except NoCredentialsError:
        print("Credentials not available")

def generate_image_from_article(content, title) -> dict:
    try:
        # Construct the image prompt from the article content
        image_prompt = f"Create a wide-angle, ultra-realistic image based on the following context and it should appears as if captured in real life. And there should be no text written on the image {content}"

        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)

        # Generate the image using DALL·E
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )

        # Retrieve the URL of the generated image
        image_url = response.data[0].url

        # Return the image URL and title
        return {"title": title, "image_url": image_url}

    except Exception as e:
        # Raise an HTTP exception with the error message in case of failure
        raise HTTPException(status_code=500, detail=str(e))

def create_article(article_data, title, images, keyword, tags):
    articleData = {
        "title": title,
        "article": article_data['seo_suggestions']['article'],
        "image_url": images,
        "keyword": keyword,
        "ai_percentage": article_data['detection_result']['data']['fakePercentage'],
        "total_words":  article_data['detection_result']['data']['textWords'],
        "seo_suggestion": article_data['seo_suggestions']['seo_suggestions'],
        "total_paragraphs": article_data['seo_suggestions']['total_paragraphs'],
        "tags": tags
    }
        
    print("Article Object is", articleData)
    # Create an Article instance from the input data
    new_article = Article(**articleData)
    
    # Add and commit the new article to the session
    db.add(new_article)
    db.commit()
    db.refresh(new_article)  # Refresh to get generated ID and other values
    
    return new_article

@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    # Call the function to generate the image
    result = generate_image_from_article(request.article_content, request.article_title)
    return result


# Optional - Endpoint to check API health
@app.get("/")
def read_root():
    return {"status": "API is up and running"}

# Get all articles
@app.get("/articles/", response_model=List[ArticleSchema])
def get_articles(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    # Query the database for articles
    articles = db.query(Article).offset(skip).limit(limit).all()

    # Ensure that fields are correctly formatted before returning them
    for article in articles:
        # Convert `seo_suggestion` to a dictionary if it's a JSON string
        if article.seo_suggestion and isinstance(article.seo_suggestion, str):
            try:
                article.seo_suggestion = json.loads(article.seo_suggestion)
            except ValueError:
                # If it can't be parsed, set it to an empty dictionary
                article.seo_suggestion = {}

        # If `seo_suggestion` is None or not a dict, ensure it's set to {}
        if article.seo_suggestion is None or not isinstance(article.seo_suggestion, dict):
            article.seo_suggestion = {}

        # Ensure that `image_url` is a list if it's a JSON string or None
        if article.image_url and isinstance(article.image_url, str):
            try:
                article.image_url = json.loads(article.image_url)
            except ValueError:
                article.image_url = []

        # If `image_url` is None or not a list, set it to an empty list
        if article.image_url is None or not isinstance(article.image_url, list):
            article.image_url = []

        # Convert datetime fields to strings for JSON serialization
        if article.created_at:
            article.created_at = article.created_at.isoformat()
        if article.updated_at:
            article.updated_at = article.updated_at.isoformat()
        if article.deleted_at:
            article.deleted_at = article.deleted_at.isoformat()

    return articles

def post_to_wp(article_id, post_date, category_id):

    article = db.query(Article).filter(Article.id == article_id).first()

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    article_data = generate_html_file(article.article, article.title, article.image_url)
    media_id = upload_image_to_wp(article.image_url[0])
    
    tag_ids = [get_or_create_tag(tag) for tag in article.tags]
    print("Tag ids", tag_ids)
    # Prepare the payload for the post
    post_data = {
        "title": article.title,
        "content": article_data,
        "status": "publish",
        "date": post_date,
        "categories": [category_id],
        "tags": tag_ids,
        "featured_media": media_id
    }

    # Use basic authentication to authenticate with the WordPress API
    response = requests.post(
        endpoint,
        json=post_data,
        auth=(wp_username, wp_password),
        headers={"Content-Type": "application/json"},
    )

    # Check if the request was successful
    if response.status_code == 201:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to post article to WordPress: {response.content}",
        )

# Define the new endpoint for posting articles to WordPress
@app.post("/post-to-wordpress/")
def post_article_to_wordpress(request: PostToWordPressRequest, db: Session = Depends(get_db)):
    # Post the article to WordPress
    response = post_to_wp(
        article_id=request.article_id,
        post_date=request.post_date,
        category_id=request.wp_category_id
    )
    print("Article uploaded")

    return {
        "message": "Article posted to WordPress successfully",
        "wordpress_response": response,
    }
    
def get_tags(text):
    print("We are in get tags", text)
    # Use a regular expression to find the tags after "Suggested Tags:"
    match = re.search(r"Tags:\s*([a-zA-Z0-9,\s-]+)", text)
    print("Match text", match)

    # Check if the match was found and extract the tags
    if match:
        tags_str = match.group(1)
        # Split the tags string into a list by separating at commas
        tags = [tag.strip() for tag in tags_str.split(",")]

    # Print the extracted tags
    print(tags)
    return tags

def upload_image_to_wp(image_url):
    download_image(image_url, wp_image_name)
    
    print(image_url, wp_image_name)
    local_image_path = "images/wp_image_name.png"
    
    headers = { "Accept": "application/json" }

    response = requests.post(
        media_endpoint,
        headers=headers,
        files={'file': open(local_image_path, 'rb')},
        auth=(wp_username, wp_password),
    )

    if response.status_code == 201:
        media_response = response.json()
        media_id = media_response["id"]  # Get the media ID from the response
        print(f"Image uploaded successfully. Media ID: {media_id}")
        return media_id
    else:
        print(f"Failed to upload image: {response.content}")
        return None
   
def get_or_create_tag(tag_name: str) -> int:
    # Define the tag endpoint
    
    # Search for the tag by name
    response = requests.get(tags_endpoint, params={"search": tag_name}, auth=(wp_username, wp_password))
    
    if response.status_code == 200:
        tags = response.json()
        # If tag exists, return its ID
        if tags:
            return tags[0]['id']
    
    # If tag doesn't exist, create it
    create_tag_response = requests.post(
        tags_endpoint,
        json={"name": tag_name},
        auth=(wp_username, wp_password),
        headers={"Content-Type": "application/json"},
    )
    
    if create_tag_response.status_code == 201:
        return create_tag_response.json()['id']
    else:
        raise HTTPException(
            status_code=create_tag_response.status_code,
            detail=f"Failed to create tag: {create_tag_response.content.decode()}",
        )
        
@app.post("/generate-and-post-article/")
async def generate_and_post_article(request: GenerateAndPostRequest):
    try:
        article_data = generate_article(request.title)
        print("article_data", article_data)
        post_to_wp(article_data["id"], request.post_date, request.wp_category_id)
        return article_data

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the article: {str(e)}")
