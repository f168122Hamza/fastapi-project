from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
import os
from pydantic import BaseModel
import re
from bs4 import BeautifulSoup
from docx import Document
from docx.shared import RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Initialize FastAPI app and load environment variables
load_dotenv()
app = FastAPI()

# Print debug information to ensure the API key is loaded correctly
print("====================HELLO JEE CHAYE PEE LO======================")
print(os.environ.get("CHATBOT_KEY"))
print(os.environ.get("PARAPHRASE_KEY"))
print(os.environ.get("ZERO_GPT_API_KEY"))

# Setup OpenAI API Key
api_key = os.environ.get("CHATBOT_KEY")
paraphrase_api_key = os.environ.get("PARAPHRASE_KEY")
zero_gpt_api_key = os.environ.get("ZERO_GPT_API_KEY")


# Input Model for request body
class ArticleRequest(BaseModel):
    title: str
    word_count: int


class ImageRequest(BaseModel):
    article_title: str
    article_content: str


# Generate an article based on a title
@app.post("/generate-article/")
async def generate_article(request: ArticleRequest):
    try:
        # ChatGPT prompt that encourages a human-like tone and discourages AI mentions
        prompt = (
            f"As a talented writer and SEO expert, write a detailed article on my topic '{request.title}'."
                "The article's length must be between 2000 words and it is a must."
                "Paragraph length must be 300 words"
                "Maximum number of paragraphs should be 7"
                "Give a SEO optimised 'Title' that complements the article"
                "Write like a human, with more spoken language. It is a must!"
                "For every new paragraph start put --- in the start"
                "No spelling or grammer mistake should be in any paragraph"
                # "2. You should hyperlink to every important term that is related to the online relevant resources you have researched. Furthermore, you should include the secondary keywords in the article and bolden them."
                # "Use semantically relevant keywords about my topic in bold format to improve semantic SEO"
                # "6. The article should help readers with a step-by-step guide where appropriate."
                # "7. Hyperlink every important term/phrase to the online relevant resource to add more context. readers to buy the product."
                # "8. The article should confidently convince"
                # "9. Format in bold style where my keywords are present."
                # "10. Format the article with h1, h2, h3, h4, and other formatting styles in an appropriate manner"
                # "Now write the publish-ready article for me, there should be no extra thing apart from the main article you wrote for me."
                "After you have written the article, you need to do the following things:"
                "Make a paragraph with heading 'SEO Suggestions'"
                "Suggest primary keyword for my content."
                "Suggest a kickass meta description within 160 characters for my content."
            )

        client = OpenAI(api_key=api_key)
        # Request completion from OpenAI using the correct method and parameters
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a article writer"},
                {"role": "user", "content": prompt},
            ],
            max_tokens= 4000,
        )

        # print("response", response.choices[0].message.content)
        answer = response.choices[0].message.content
        # Extract and return the article content from the response
        formatted_article = format_article(answer)
        print("answer", formatted_article)

        # paraphrased_article = await paraphrase_article_prepostseo(formatted_article)
        
        seo_suggestions = clean_article(formatted_article)
        print("SEO suggestions", seo_suggestions)
        detection_result = detect_ai_content(formatted_article)
        if detection_result['success'] and detection_result['data']['fakePercentage'] < 40 and detection_result['data']['textWords'] > 1000 :
            print("Article is good to go")
        else:
            print("Regenerate Article as percentage is high")
            # generate_article()

        # print("detection_result", detection_result)

        return {
            "article": formatted_article,
            "ai_detection": detection_result
        }

    except Exception as e:
        # Return an error message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))

ZERO_GPT_API_URL = "https://api.zerogpt.com/api/detect/detectText"

def detect_ai_content(text):
    # Ensure the text is not empty or None
    if not text or text.strip() == "":
        return {"error": "Input text cannot be empty or None"}

    headers = {
        'ApiKey': zero_gpt_api_key,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "text": text,
        "input_text": text  # Ensure this field has a valid non-empty string
    })

    try:
        # Synchronous POST request to the ZeroGPT API
        response = requests.post(ZERO_GPT_API_URL, headers=headers, data=payload)
        response.raise_for_status()  # Check if the request was successful
        result = response.json()
        return result
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
    except Exception as err:
        print(f"An error occurred: {err}")
        return {"error": str(err)}
        
async def paraphrase_article_prepostseo(text: str) -> str:
    try:
        # PrepostSEO API endpoint
        api_url = "https://www.prepostseo.com/apis/checkparaphrase"
        
        # Payload for the POST request
        payload = {
            'key': paraphrase_api_key,  # Your PrepostSEO API key
            'data': text,  # The text to be paraphrased
            'lang': 'en',  # Language: 'en' for English
            'mode': 'Advanced'  # Paraphrasing mode: Simple, Advanced, Fluency, Creative (optional)
        }

        # Sending POST request to PrepostSEO API
        response = requests.post(api_url, data=payload)

        if response.status_code == 200:
            # Extract the paraphrased text from the response
            response_text = response.content.decode('utf-8-sig')
            # print("=======================Simple Response========================")
            # print("Response:", response_text)
            paraphrased_text = json.loads(response_text).get('paraphrasedContent')
            # print("=======================Paraphrased Without Article========================")
            # print("Simple Article:", text)
            # print("=======================Paraphrased Article========================")
            print("Paraphrased Article:", paraphrased_text)
            return paraphrased_text
        else:
            raise Exception(f"Failed to paraphrase text. Status code: {response.status_code}, Response: {response.text}")

    except Exception as e:
        raise Exception(f"Error while paraphrasing: {str(e)}")

def format_article(text):
    # Remove special characters and extra whitespaces
    cleaned_text = re.sub(r'[^A-Za-z0-9,.?!;:\-\(\)\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Format the article into paragraphs and sections
    formatted_text = cleaned_text.replace(' ## ', '\n\n## ')
    formatted_text = formatted_text.replace(' ### ', '\n\n### ')
    formatted_text = formatted_text.replace(' - ', '\n\n- ')
    formatted_text = formatted_text.replace(' 1. ', '\n\n1. ')
    
    return formatted_text

# Function to add HTML-styled text to the document
def add_styled_text(paragraph, text, bold=False, italic=False, color=None):
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
        
def clean_article(response):
    # Remove inline styles from the response
    cleaned_answer = re.sub(r'style=\"[^\"]*\"', '', response)
    
    # Insert a new paragraph tag where '---' appears
    cleaned_answer = re.sub(r'---', '</p><p>', cleaned_answer)
    # cleaned_answer = re.sub(r'. - ', '', cleaned_answer)
    
    
    seo_suggestions = ''
    seo_match = re.search(r'(SEO Suggestions.*?)(?=---|<\/p>|$)', cleaned_answer, re.DOTALL)
    if seo_match:
        seo_suggestions = seo_match.group(1).strip()
        cleaned_answer = cleaned_answer.replace(seo_suggestions, '')

    total_paragraphs = cleaned_answer.count('<p>') + 1

    # Define the HTML content with the cleaned and modified answer
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API Response Display</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f4;
            }}
            .content {{
                background-color: #fff;
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <h1>API Response Content</h1>
        <div class="content">
            <p>{cleaned_answer}</p>
        </div>
    </body>
    </html>
    """

    # Write the HTML content to an HTML file
    with open("api_response.html", "w") as html_file:
        html_file.write(html_content)

    print("HTML file created successfully: api_response.html")
    return { "seo_suggestions": seo_suggestions, "total_paragraphs": total_paragraphs }

    
@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    try:
        # Use article content as a prompt to generate an image using DALL-E
        image_prompt = f"Create an image that visually represents the following article: {request.article_content}"

        client = OpenAI(api_key=api_key)
        # Generate the image using DALL·E
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )
        # image_response = openai.Image.create(
        #     prompt=image_prompt,
        #     n=1,  # Number of images to generate
        #     size="1024x1024"  # Image resolution (other options: "256x256", "512x512")
        # )

        # Extract the URL of the generated image
        image_url = response.data[0].url

        return {"title": request.article_title, "image_url": image_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional - Endpoint to check API health
@app.get("/")
def read_root():
    return {"status": "API is up and running"}
