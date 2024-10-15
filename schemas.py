from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

# Define the schema for an article response
class ArticleSchema(BaseModel):
    id: int
    title: str
    article: str
    image_url: Optional[List[str]] = None  # List of image URLs (optional)
    keyword: str
    ai_percentage: float
    total_words: int
    seo_suggestion: Optional[Dict[str, str]] = None  # SEO suggestions (optional and should be a dictionary)
    total_paragraphs: int
    tags: Optional[Dict[str, str]] = None
    
    # Change `created_at` and `updated_at` to `datetime` type
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None  # Optional datetime for soft-deleted items

    class Config:
        orm_mode = True  # Enable compatibility with SQLAlchemy models
