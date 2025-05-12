from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict # Make sure Optional is imported
import json
import os
from dotenv import load_dotenv # Make sure load_dotenv is called if you use a .env locally
from jose import JWTError, jwt # Add jwt here
from passlib.context import CryptContext
import json
from fastapi.staticfiles import StaticFiles
import os # Make sure os is imported
import logging # Make sure logging is imported

logger = logging.getLogger("api") # Ensure logger is defined

load_dotenv() # Load environment variables from .env file if present

# Import your modules
from db import get_db, engine, Base
from models import User, Recording
from realtime_emotion import RealtimeEmotionDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI()

# Get allowed origins from environment variable, split by comma if multiple
# Default to localhost:3000 for local dev if variable not set
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the origins list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# JWT settings - Read from environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key-for-local-dev-only") # Provide a default ONLY for local dev if needed
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")) # Default to 60 minutes

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # This line should now work

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# Initialize detector
logger.info("Initializing LightweightDetector")
detector = RealtimeEmotionDetector()
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

# Helper function to create access token
# Change 'timedelta | None' to 'Optional[timedelta]'
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) # Use env var
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
@app.post("/api/register")
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

@app.post("/api/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Login attempt for username: {form_data.username}") # Log username
    # Consider logging password length or a hash for debugging, but NOT the plain password in production
    # logger.info(f"Password received (length): {len(form_data.password)}")

    user = db.query(User).filter(User.username == form_data.username).first()

    if not user:
        logger.warning(f"Login failed: User '{form_data.username}' not found.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.password_hash):
        logger.warning(f"Login failed: Password verification failed for user '{form_data.username}'.")
        # You might want to log the received password's hash vs stored hash for deeper debugging if issues persist
        # received_hash_attempt = get_password_hash(form_data.password) # Don't log this directly if sensitive
        # logger.debug(f"Stored hash: {user.password_hash}, Attempted password (if hashed for compare): {received_hash_attempt}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"User '{form_data.username}' authenticated successfully.")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Your existing endpoints with authentication
@app.post("/api/start_recording/")
async def start_recording(current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Starting recording for user {current_user.username}")
        detector.start_recording()
        return {"message": "Recording started"}
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop_recording/")
async def stop_recording(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        logger.info(f"Stopping recording for user {current_user.username}")
        detector.stop_recording()
        
        analysis = detector.analyze_emotions()
        face_count = analysis.get('face_count', 0)
        logger.info(f"Analysis result summary: {face_count} faces detected over {analysis.get('duration', 0):.2f} seconds")
        
        # Save multi-face analysis to database
        new_recording = Recording(
            user_id=current_user.id,
            analysis_data=json.dumps(analysis),
            timestamp=datetime.utcnow()
        )
        
        db.add(new_recording)
        db.commit()
        logger.info(f"Recording saved to database with ID: {new_recording.id}")
        
        return analysis
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recording history endpoints
@app.get("/api/recording_history")
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

@app.get("/api/recording/{recording_id}")
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

@app.delete("/api/recording/{recording_id}")
async def delete_recording(recording_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recording = db.query(Recording).filter(
        Recording.id == recording_id, 
        Recording.user_id == current_user.id
    ).first()
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found or access denied")
    
    db.delete(recording)
    db.commit()
    
    return {"message": "Recording successfully deleted"}

# Add these imports if not already present
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict

# Create data models for the request/response
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class FaceEmotion(BaseModel):
    face_location: Tuple[int, int, int, int]  # Make sure this matches what's returned
    dominant_emotion: str
    emotion_scores: Dict[str, float]

@app.post("/api/detect_emotion")
async def detect_emotion(request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- Temporary Debugging: Save frame ---
        debug_save_path = "debug_frame.jpg"
        cv2.imwrite(debug_save_path, frame)
        print(f"Debug: Saved incoming frame to {debug_save_path}")
        # --- End Debugging ---

        # Process the frame
        frame_with_emotions = detector.process_frame(frame)
        
        # Get emotion data
        emotion_data = detector.current_emotion_data if hasattr(detector, "current_emotion_data") else []
        
        # Add a debug message if recording
        is_recording = detector.recording if hasattr(detector, "recording") else False
        if is_recording:
            # Count total records across all tracked faces
            total_records = sum(len(records) for records in detector.face_records.values()) if hasattr(detector, "face_records") else 0
            print(f"Processed frame during recording. Total records: {total_records}")
        
        # Ensure all numpy types are converted before returning
        cleaned_emotion_data = convert_numpy_types(emotion_data)

        return {
            "status": "success",
            "emotions": cleaned_emotion_data,
            "is_recording": is_recording
        }
    except Exception as e:
        import traceback
        logger.error(f"Error in detect_emotion endpoint: {str(e)}")
        logger.error(traceback.format_exc()) # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error in detect_emotion: {str(e)}")

# Add these imports if not already present
from fastapi import File, UploadFile
import io
import cv2
import numpy as np

# Add this new endpoint alongside your other endpoints
@app.post("/api/detect_emotion_from_image")
async def detect_emotion_from_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the image and detect faces/emotions using the detector
        processed_data = detector.process_uploaded_image(image)
        
        return {
            "status": "success",
            "emotions": processed_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- API Routes ---
# IMPORTANT: Mount all your existing API routes *before* mounting the static files
# Example: app.include_router(your_api_router, prefix="/api")
# OR define individual routes like @app.post("/token"), @app.post("/register"), etc.
# Make sure all API endpoints start with a prefix like /api/ or /v1/ if you mount static files at root '/'

# --- Mount Static Files (React Frontend Build) ---
# Conditionally mount static files only if the directory exists
# This allows running locally without the 'static' folder crashing the app
static_dir_path = "static" # Relative path from where api.py runs
if os.path.isdir(static_dir_path):
    app.mount("/", StaticFiles(directory=static_dir_path, html=True), name="static")
    logger.info(f"Serving static files from directory: {static_dir_path}")
else:
    logger.warning(f"Static files directory '{static_dir_path}' not found. Frontend will not be served by this instance (normal for local dev without frontend build).")