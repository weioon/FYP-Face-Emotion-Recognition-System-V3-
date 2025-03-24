from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
import datetime
from db import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}  # Add this line
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password_hash = Column(String(128))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    recordings = relationship("Recording", back_populates="user")

class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = {'extend_existing': True}  # Add this line
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    analysis_data = Column(Text)  # Store JSON as text
    
    user = relationship("User", back_populates="recordings")