from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import openai
import os
from pydantic import BaseModel

# Initialize FastAPI app and load environment variables
load_dotenv()
app = FastAPI()

# Print debug information to ensure the API key is loaded correctly
print("====================HELLO JEE CHAYE PEE LO======================")
print(os.environ.get("CHATBOT_KEY"))

# Setup OpenAI API Key
api_key = os.environ.get("CHATBOT_KEY")


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
            f"Write a detailed article based on the title '{request.title}'. "
            "The tone should be friendly and engaging, avoiding robotic language. "
            "Ensure the article is original, human-like, and does not mention that it was written using AI tools. "
            "Keep the article concise, informative, and plagiarism-free."
            "Make sure the AI generated context should be less than 5 percent if i give it to any AI detecting tool currently it is 100 percent AI generated"
        )

        client = OpenAI(api_key=api_key)
        # Request completion from OpenAI using the correct method and parameters
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        print("response", response)
        print("response", response.choices[0].message.content)
        answer = response.choices[0].message.content
        # Extract and return the article content from the response

        return {"answer": answer}

    except Exception as e:
        # Return an error message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))


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
