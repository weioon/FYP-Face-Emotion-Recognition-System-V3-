from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
import json
import logging
from jose import jwt
from passlib.context import CryptContext
import base64
import numpy as np
import cv2

# Import your modules
from db import get_db, engine, Base
from models import User, Recording
from lightweight_detector import LightweightDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Create FastAPI app
app = FastAPI()

# Configure CORS - IMPORTANT!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# JWT settings
SECRET_KEY = "your-secret-key"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize detector
logger.info("Initializing LightweightDetector")
detector = LightweightDetector()
logger.info("LightweightDetector initialized")

# Create database tables
Base.metadata.create_all(bind=engine)

# User registration model
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Log the token for debugging (remove in production)
        logger.info(f"Validating token: {token[:10]}...")
        
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            raise credentials_exception
            
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            logger.error(f"User not found: {username}")
            raise credentials_exception
            
        return user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

# Authentication endpoints
@app.post("/register")
async def register_user(user: UserRegister, db: Session = Depends(get_db)):
    logger.info(f"Registration request received for: {user.username}")
    
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    
    if (existing_user):
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, password_hash=hashed_password)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Your existing endpoints with authentication
@app.post("/start_recording/")
async def start_recording(current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Starting recording for user {current_user.username}")
        detector.start_recording()
        return {"message": "Recording started"}
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_recording/")
async def stop_recording(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        logger.info(f"Stopping recording for user {current_user.username}")
        detector.stop_recording()
        
        analysis = detector.analyze_emotions()
        logger.info(f"Analysis result summary: Duration={analysis.get('duration', 0)}, Dominant emotion={analysis.get('dominant_emotion', 'unknown')}")
        
        # Always save to database, even with minimal data
        new_recording = Recording(
            user_id=current_user.id,
            analysis_data=json.dumps(analysis),
            timestamp=datetime.utcnow()  # Explicitly set UTC time
        )
        
        db.add(new_recording)
        db.commit()
        logger.info(f"Recording saved to database with ID: {new_recording.id}")
        
        # Return the analysis for immediate display
        return analysis
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recording history endpoints
@app.get("/recording_history")
async def get_recording_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recordings = db.query(Recording).filter(Recording.user_id == current_user.id).all()
    
    results = []
    for rec in recordings:
        results.append({
            "id": rec.id,
            "timestamp": rec.timestamp.isoformat(),
            "analysis_data": json.loads(rec.analysis_data)
        })
        
    return results

@app.get("/recording/{recording_id}")
async def get_recording(recording_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recording = db.query(Recording).filter(
        Recording.id == recording_id, 
        Recording.user_id == current_user.id
    ).first()
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found or access denied")
        
    return {
        "id": recording.id,
        "timestamp": recording.timestamp.isoformat(),
        "analysis_data": json.loads(recording.analysis_data)
    }

# Add these imports if not already present
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from typing import Optional, List

# Create data models for the request/response
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class FaceEmotion(BaseModel):
    face_location: List[int]  # [x1, y1, x2, y2]
    dominant_emotion: str
    emotion_scores: Optional[dict] = None

class EmotionResponse(BaseModel):
    status: str
    emotions: List[FaceEmotion] = []

@app.post("/detect_emotion", response_model=EmotionResponse)
async def detect_emotion(request: ImageRequest, current_user: User = Depends(get_current_user)):
    try:
        # Log the request (but not the full image data)
        logger.info("Received emotion detection request")
        
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # Use the lightweight detector instead
        emotions = detector.detect_emotions_lightweight(frame)
        
        return {
            "status": "success",
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Error in /detect_emotion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")