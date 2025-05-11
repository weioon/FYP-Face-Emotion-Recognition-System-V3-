import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv() # Keep this if you use a .env file locally

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL") # Read from environment variable

# Default to SQLite if DATABASE_URL is not set
if not SQLALCHEMY_DATABASE_URL:
    print("DATABASE_URL not set, defaulting to SQLite: ./sql_app.db")
    SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db" # Default SQLite path

# Handle PostgreSQL scheme if needed
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # Needed for SQLite
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Add this function definition
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()