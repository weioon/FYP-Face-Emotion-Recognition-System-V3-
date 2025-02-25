from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from realtime_model import RealtimeEmotionDetector

app = FastAPI()
detector = RealtimeEmotionDetector()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_emotions(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame
    processed_frame = detector.process_frame(frame)
    
    # Encode processed frame to base64
    _, img_buffer = cv2.imencode('.jpg', processed_frame)
    return {"image": base64.b64encode(img_buffer).decode('utf-8')}