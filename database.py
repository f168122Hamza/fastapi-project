from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define MySQL database URL
DATABASE_URL = "mysql+mysqlconnector://root:root@localhost:3311/wp-automation"

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a sessionmaker factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
