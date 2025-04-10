from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import base64
import os
import time
from io import BytesIO

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a model for request validation
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

# Initialize models
yolo_model = YOLO('yolov8n.pt')

class RealtimeEmotionDetector:
    def __init__(self, debug_mode=False):
        # Use a single consistent variable name for emotion records
        self.emotion_records = []
        self.recording = False
        self.start_time = None
        self.end_time = None
        self.current_emotion_data = []
        self.debug_mode = debug_mode
        print("RealtimeEmotionDetector initialized")
    
    def start_recording(self):
        """Start a new recording session"""
        self.recording = True
        self.start_time = time.time()
        self.emotion_records = []  # Clear previous records
        print(f"Recording started at {self.start_time}")
        return True
    
    def stop_recording(self):
        """Stop the current recording session"""
        self.recording = False
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        print(f"Recording stopped after {duration:.2f}s with {len(self.emotion_records)} emotions recorded")
        return True
    
    def detect_emotions_in_frame(self, frame_data):
        """Detect emotions in a single frame - used by the detect_emotion endpoint"""
        frame = self.decode_image(frame_data)
        processed_frame = self.process_frame(frame)
        
        # Return the current emotions that were just detected
        return self.current_emotion_data
    
    def process_frame(self, frame):
        """Process a single frame and detect emotions"""
        try:
            # Make a copy to avoid modifying the original
            img = frame.copy()
            
            # Resize image for faster processing
            scale = 0.5
            small_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
            emotions_detected = []
            face_detected = False
            
            try:
                # Print frame processing timestamp for debugging
                current_time = time.time()
                elapsed = current_time - self.start_time if self.recording and self.start_time else 0
                print(f"Processing frame at {elapsed:.2f}s from start")
                
                results = DeepFace.analyze(
                    small_frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                
                # Normalize results
                if not isinstance(results, list):
                    results = [results]
                
                for result in results:
                    if 'emotion' in result:
                        face_detected = True
                        # Extract face location
                        x = int(result['region']['x'] / scale) if 'region' in result else 0
                        y = int(result['region']['y'] / scale) if 'region' in result else 0
                        w = int(result['region']['w'] / scale) if 'region' in result else 100
                        h = int(result['region']['h'] / scale) if 'region' in result else 100
                        
                        # Extract emotions
                        emotions_dict = result['emotion']
                        dominant_emotion = result['dominant_emotion']
                        
                        # Format emotion scores
                        emotion_scores = {k: float(v) for k, v in emotions_dict.items()}
                        
                        emotions_detected.append({
                            "face_location": [x, y, x+w, y+h],
                            "dominant_emotion": dominant_emotion,
                            "emotion_scores": emotion_scores
                        })
                        
                        # Draw rectangle for visualization
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Record emotion data when recording is active
                        if self.recording:
                            timestamp = current_time - self.start_time if self.start_time else 0
                            
                            # Always make first letter uppercase for consistency
                            emotion_name = dominant_emotion.capitalize()
                            
                            # Add the record to emotion_records
                            self.emotion_records.append({
                                "timestamp": timestamp,
                                "emotion": emotion_name,
                                "score": emotion_scores.get(dominant_emotion.lower(), 0)
                            })
                            print(f"Recorded emotion: {emotion_name} at {timestamp:.2f}s - Total: {len(self.emotion_records)}")
            
            except Exception as e:
                print(f"DeepFace error: {e}")
            
            # Record neutral emotion if no face detected during recording
            if self.recording and not face_detected:
                timestamp = time.time() - self.start_time if self.start_time else 0
                print(f"No face detected - recording neutral at {timestamp:.2f}s")
                self.emotion_records.append({
                    "timestamp": timestamp,
                    "emotion": "Neutral",
                    "score": 100.0
                })
            
            # Store current emotion data for API access
            self.current_emotion_data = emotions_detected
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
            traceback.print_exc()
            return frame
    
    def decode_image(self, base64_image):
        """Decode a base64 image to a CV2 image (numpy array)"""
        try:
            # For pure base64 string (without data:image prefix)
            if isinstance(base64_image, np.ndarray):
                # If it's already a numpy array, return it directly
                return base64_image
                
            # Handle string or byte input
            if isinstance(base64_image, str):
                # Remove data:image prefix if present
                if ',' in base64_image:
                    base64_image = base64_image.split(',')[1]
                    
                image_data = base64.b64decode(base64_image)
            elif isinstance(base64_image, bytes):
                image_data = base64_image
            else:
                raise ValueError("Unsupported image format")
                
            # Convert to numpy array and then to CV2 image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def analyze_emotions(self):
        """Analyze recorded emotion data and provide insights"""
        print(f"Starting emotion analysis with {len(self.emotion_records)} records")
        
        # Calculate duration
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Basic error handling - return default structure if no data
        if not self.emotion_records or len(self.emotion_records) < 3:
            print("Not enough emotion records for analysis")
            return {
                "error": "No emotions were recorded. Please try again and make sure your face is visible.",
                "duration": duration,
                "stats": {"Neutral": 100.0},
                "emotion_journey": {
                    "beginning": {"Neutral": 100.0},
                    "middle": {"Neutral": 100.0},
                    "end": {"Neutral": 100.0}
                }
            }
        
        try:
            # Count emotions
            emotion_counts = {}
            for record in self.emotion_records:
                emotion = record["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            total_records = len(self.emotion_records)
            
            # Calculate percentages
            emotion_percentages = {
                emotion: (count / total_records) * 100
                for emotion, count in emotion_counts.items()
            }
            
            # Get dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # Split the timeline into three parts
            beginning_idx = max(1, int(total_records * 0.25))
            middle_idx = max(1, int(total_records * 0.5)) 
            end_idx = max(1, int(total_records * 0.75))
            
            beginning_records = self.emotion_records[:beginning_idx]
            middle_records = self.emotion_records[beginning_idx:end_idx]
            end_records = self.emotion_records[end_idx:]
            
            # Analyze each segment
            beginning_emotions = {}
            for record in beginning_records:
                emotion = record["emotion"]
                beginning_emotions[emotion] = beginning_emotions.get(emotion, 0) + 1
            
            middle_emotions = {}
            for record in middle_records:
                emotion = record["emotion"]
                middle_emotions[emotion] = middle_emotions.get(emotion, 0) + 1
                
            end_emotions = {}
            for record in end_records:
                emotion = record["emotion"]
                end_emotions[emotion] = end_emotions.get(emotion, 0) + 1
            
            # Calculate percentages for each segment
            beginning_percentages = {
                emotion: (count / len(beginning_records)) * 100
                for emotion, count in beginning_emotions.items()
            }
            
            middle_percentages = {
                emotion: (count / len(middle_records)) * 100
                for emotion, count in middle_emotions.items()
            }
            
            end_percentages = {
                emotion: (count / len(end_records)) * 100
                for emotion, count in end_emotions.items()
            }
            
            # Create interpretation and recommendations based on emotion journey
            interpretation = self._generate_interpretation(beginning_percentages, middle_percentages, end_percentages)
            recommendations = self._generate_recommendations(emotion_percentages, dominant_emotion)
            
            # Format for analysis result
            analysis_result = {
                "duration": duration,
                "dominant_emotion": dominant_emotion,
                "stats": emotion_percentages,
                "significant_emotions": [
                    {"emotion": emotion, "percentage": percentage}
                    for emotion, percentage in emotion_percentages.items()
                    if percentage > 5.0  # Only include emotions above 5%
                ],
                "emotion_journey": {
                    "beginning": beginning_percentages,
                    "middle": middle_percentages,
                    "end": end_percentages
                },
                "interpretation": interpretation,
                "educational_recommendations": recommendations
            }
            
            print(f"Analysis completed successfully with dominant emotion: {dominant_emotion}")
            return analysis_result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in analyze_emotions: {str(e)}")
            
            # Return error but with consistent data structure
            return {
                "error": f"Analysis error: {str(e)}",
                "duration": duration,
                "stats": {"Neutral": 100.0},
                "emotion_journey": {
                    "beginning": {"Neutral": 100.0},
                    "middle": {"Neutral": 100.0},
                    "end": {"Neutral": 100.0}
                }
            }

    def _generate_interpretation(self, beginning, middle, end):
        """Generate interpretation based on emotion journey"""
        try:
            # Simple interpretation based on emotion shifts
            beginning_emotion = max(beginning.items(), key=lambda x: x[1])[0] if beginning else "Neutral"
            middle_emotion = max(middle.items(), key=lambda x: x[1])[0] if middle else "Neutral"
            end_emotion = max(end.items(), key=lambda x: x[1])[0] if end else "Neutral"
            
            interpretation = f"Your emotional journey started with {beginning_emotion}, "
            interpretation += f"moved to {middle_emotion} in the middle, "
            interpretation += f"and ended with {end_emotion}. "
            
            if beginning_emotion == middle_emotion == end_emotion:
                interpretation += "Your emotional state remained consistent throughout the session."
            elif beginning_emotion != end_emotion:
                interpretation += f"Your emotion changed from {beginning_emotion} to {end_emotion}, which suggests a significant emotional shift during the session."
            
            return interpretation
        except Exception as e:
            print(f"Error generating interpretation: {e}")
            return "Unable to generate interpretation due to insufficient data."

    def _generate_recommendations(self, emotions, dominant_emotion):
        """Generate educational recommendations based on emotions"""
        try:
            recommendations = []
            
            if dominant_emotion == "Happy" or dominant_emotion == "Surprise":
                recommendations.append("Your positive engagement is excellent! Consider tackling more challenging material.")
                recommendations.append("Try explaining concepts to others to reinforce your understanding.")
            elif dominant_emotion == "Sad":
                recommendations.append("Consider taking short breaks to refresh your mind.")
                recommendations.append("Try connecting difficult concepts to topics you enjoy.")
            elif dominant_emotion == "Fear" or dominant_emotion == "Disgust":
                recommendations.append("Break down complex topics into smaller, manageable parts.")
                recommendations.append("Consider group study sessions to gain different perspectives.")
            elif dominant_emotion == "Angry":
                recommendations.append("Take a short break and return with a fresh perspective.")
                recommendations.append("Try a different learning approach or resource for this topic.")
            else:  # Neutral
                recommendations.append("Consider more interactive learning methods to increase engagement.")
                recommendations.append("Try setting specific learning goals to measure progress.")
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Unable to generate personalized recommendations."]

detector = RealtimeEmotionDetector()

@app.post("/detect_emotion")
async def detect_emotion(request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        result_frame = detector.process_frame(frame)
        
        # Extract emotion data (assuming your process_frame returns this information)
        # You'll need to modify your process_frame to return emotion data
        emotion_data = detector.current_emotion_data if hasattr(detector, "current_emotion_data") else {}
        
        return {
            "status": "success",
            "emotions": emotion_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/start_recording")
async def start_recording():
    detector.start_recording()
    return {"status": "recording_started"}

@app.post("/stop_recording")
async def stop_recording():
    detector.stop_recording()
    analysis_result = detector.analyze_emotions()
    return analysis_result

def main():
    detector = RealtimeEmotionDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Controls:")
    print("- Press 'r' to start/stop recording")
    print("- Press 's' to adjust the sensitivity of Neutral→Sad classification")
    print("- Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        result_frame = detector.process_frame(frame)
        
        # Add recording status to frame
        if detector.recording:
            rec_text = "RECORDING"
            cv2.putText(result_frame, rec_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add current sensitivity setting
        sensitivity_text = f"Neutral->Sad threshold: {detector.sad_neutral_ratio_threshold:.1f}"
        cv2.putText(result_frame, sensitivity_text, (10, frame.shape[0] - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Realtime Emotion Detection', result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if detector.recording:
                detector.stop_recording()
                analysis_result = detector.analyze_emotions()
                
                # Display enhanced detailed analysis
                print("\n========== COMPREHENSIVE EMOTION ANALYSIS REPORT ==========")
                print(f"Recording duration: {analysis_result['duration']:.2f} seconds")
                
                print("\n📊 EMOTION DISTRIBUTION:")
                for emotion, percentage in analysis_result['stats'].items():
                    print(f"- {emotion}: {percentage:.1f}%")
                
                print("\n🔄 EMOTIONAL JOURNEY:")
                print("Beginning stage:")
                for emotion, pct in analysis_result['emotion_journey']['beginning'].items():
                    print(f"  - {emotion}: {pct:.1f}%")
                print("Middle stage:")
                for emotion, pct in analysis_result['emotion_journey']['middle'].items():
                    print(f"  - {emotion}: {pct:.1f}%")
                print("End stage:")
                for emotion, pct in analysis_result['emotion_journey']['end'].items():
                    print(f"  - {emotion}: {pct:.1f}%")
                
                print("\n📝 INTERPRETATION:")
                print(analysis_result['interpretation'])
                
                print("\n⚠️ SIGNIFICANT EMOTIONAL SHIFTS:")
                if analysis_result['significant_shifts']:
                    for shift in analysis_result['significant_shifts']:
                        print(f"- {shift}")
                else:
                    print("- No significant emotional shifts detected")
                
                print("\n🎓 EDUCATIONAL RECOMMENDATIONS:")
                for rec in analysis_result['educational_recommendations']:
                    print(f"- {rec}")
                
                print("===========================================================\n")
            else:
                detector.start_recording()
        elif key == ord('s'):
            # Adjust sensitivity
            new_value = float(input("Enter new Neutral->Sad threshold (0.1-1.0, lower is more sensitive): "))
            detector.calibrate_emotion_sensitivity(new_value)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)