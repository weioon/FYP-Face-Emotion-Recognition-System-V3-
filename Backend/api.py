from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import base64
import os
from realtime_emotion import RealtimeEmotionDetector
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str

class SensitivityRequest(BaseModel):
    sad_neutral_ratio: float

detector = RealtimeEmotionDetector()

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    try:
        print("Received request to /process_frame/")
        # Read the image data
        contents = await file.read()
        print(f"Image data size: {len(contents)} bytes")
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        print("Image decoded successfully, shape:", img.shape)
        # Process with detector
        print("Processing frame with RealtimeEmotionDetector...")
        result_frame = detector.process_frame(img)
        if result_frame is None:
            print("Frame processing returned None")
            raise HTTPException(status_code=500, detail="Frame processing returned None")

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print("Frame processed and encoded to base64, size:", len(img_base64))
        
        return {"image": f"data:image/jpeg;base64,{img_base64}"}
    except Exception as e:
        print(f"Error in /process_frame/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/start_recording/")
async def start_recording():
    try:
        print("Starting recording...")
        detector.start_recording()
        return {"message": "Recording started"}
    except Exception as e:
        print(f"Error in /start_recording/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting recording: {str(e)}")

@app.post("/stop_recording/")
async def stop_recording():
    try:
        print("Stopping recording...")
        detector.stop_recording()
        analysis = detector.analyze_emotions()
        if analysis is None:
            print("Analysis returned None")
            raise ValueError("Analysis returned None")
        print(f"Analysis result: {analysis}")
        return analysis
    except Exception as e:
        print(f"Error in /stop_recording/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping recording: {str(e)}")

@app.post("/adjust_sensitivity/")
async def adjust_sensitivity(request: SensitivityRequest):
    try:
        print(f"Adjusting sensitivity to {request.sad_neutral_ratio}")
        detector.calibrate_emotion_sensitivity(request.sad_neutral_ratio)
        return {"message": f"Sensitivity set to {request.sad_neutral_ratio}"}
    except Exception as e:
        print(f"Error in /adjust_sensitivity/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adjusting sensitivity: {str(e)}")

@app.post("/api/detect-emotion")
async def detect_emotion(image_request: ImageRequest):
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(image_request.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame with the detector
        detector = RealtimeEmotionDetector()
        processed_frame = detector.process_frame(img)
        
        # Create response with emotion data
        # Convert processed frame back to base64 for web display
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "processed_image": processed_image,
            "emotion_records": detector.emotion_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)