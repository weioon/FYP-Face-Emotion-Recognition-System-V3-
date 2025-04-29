import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv() # Keep this if you use a .env file locally

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL") # Read from environment variable

# Handle PostgreSQL scheme if needed (DigitalOcean often provides 'postgres://')
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set") # This line raises the error

engine = create_engine(SQLALCHEMY_DATABASE_URL)
# If using SQLite locally for testing, you might need conditional logic,
# but for deployment, the DATABASE_URL will be PostgreSQL.
# Example for local SQLite fallback (optional):
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # Only needed for SQLite
# )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Add this function definition
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()