from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import os
from pydantic import BaseModel

# Initialize FastAPI app
load_dotenv()
app = FastAPI()
client = OpenAI()
# Setup OpenAI API Key
client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

# Input Model
class ArticleRequest(BaseModel):
    title: str
    word_count: int

# Generate an article based on a title
@app.post("/generate-article/")
async def generate_article(request: ArticleRequest):
    try:
        # ChatGPT prompt that encourages a human-like tone and discourages AI mentions
        prompt = (f"Write a detailed article based on the title '{request.title}'. "
                  "The tone should be friendly and engaging, avoiding robotic language. "
                  "Ensure the article is original, human-like, and does not mention that it was written using AI tools. "
                  "Keep the article concise, informative, and plagiarism-free.")

        # Request completion from OpenAI
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=request.word_count
        )

        # Extract and return the article content
        article_content = response.choices[0].text.strip()
        
        return {"title": request.title, "article": article_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional - Endpoint to check API health
@app.get("/")
def read_root():
    return {"status": "API is up and running"}
