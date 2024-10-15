from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Float, JSON
from sqlalchemy.sql import func
from database import Base

# Define the Article model
class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    
    # Large text field to store up to 2000 words
    article = Column(Text, nullable=False)
    
    # JSON field to store an array of image URLs
    image_url = Column(JSON, nullable=True)
    
    # Additional fields
    keyword = Column(String(255), nullable=False)
    ai_percentage = Column(Float, nullable=False)
    total_words = Column(Integer, nullable=False)
    seo_suggestion = Column(JSON, nullable=True)
    total_paragraphs = Column(Integer, nullable=False)
    tags = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP, nullable=True)  # Nullable for soft deletion
