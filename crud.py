from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List
from models import Article
from database import get_db

app = FastAPI()

# Create an article
# def create_article(db: Session, article: schemas.ArticleCreate):
#     db_article = models.Article(title=article.title, content=article.content, image_url=article.image_url)
#     db.add(db_article)
#     db.commit()
#     db.refresh(db_article)
#     return db_article

# # Get an article by ID
# def get_article(db: Session, article_id: int):
#     return db.query(models.Article).filter(models.Article.id == article_id).first()

