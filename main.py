from fastapi import FastAPI, HTTPException # type: ignore
from dotenv import load_dotenv # type: ignore
from openai import OpenAI # type: ignore
import os
from pydantic import BaseModel # type: ignore
# from utils import generate_description # type: ignore

# Initialize FastAPI app
load_dotenv()
app = FastAPI()
# client = OpenAI()
print("====================HELLO JEE CHAYE PEE LO======================")
print(os.environ.get("OPENAI_API_KEY"))
# Setup OpenAI API Key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# # Input Model
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
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
