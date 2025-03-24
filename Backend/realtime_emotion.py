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
        self.recording = False
        self.start_time = None
        self.emotion_data = []
        self.current_emotion_data = []
        self.debug_mode = debug_mode
        self.yolo_model = YOLO('yolov8n.pt')
        self.sad_neutral_ratio_threshold = 0.3  # Default value
        self.emotion_records = []
        self.end_time = None
        # Threshold for considering an emotion significant
        self.significance_threshold = 10.0  # percentage
        # Threshold for reclassifying neutral as sad
        self.sad_neutral_ratio_threshold = 0.4  # If sad score is 40% of neutral, classify as sad

    def start_recording(self):
        self.emotion_records = []
        self.recording = True
        self.start_time = time.time()
        print("Recording started...")

    def stop_recording(self):
        self.recording = False
        self.end_time = time.time()
        print("Recording stopped.")

    def analyze_emotions(self):
        """Analyze recorded emotion data and provide insights"""
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # If no emotions were recorded, return a helpful message
        if not self.emotion_data or len(self.emotion_data) < 3:
            print("No emotions were recorded during the session.")
            return {
                "error": "No emotions were recorded. Please try again and make sure your face is visible.",
                "duration": duration,
                "stats": {"neutral": 100.0},  # Provide fallback data for frontend
                "emotion_journey": {
                    "beginning": {"neutral": 100.0},
                    "middle": {"neutral": 100.0},
                    "end": {"neutral": 100.0}
                }
            }
            
        # Continue with regular analysis...
        # (rest of your existing analyze_emotions code)
        if not self.emotion_records:
            return "No emotions recorded."

        # Calculate basic stats
        emotion_counts = {}
        for record in self.emotion_records:
            emotion = record['emotion']
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1

        total_records = len(self.emotion_records)
        total_duration = self.end_time - self.start_time
        
        # Calculate percentages
        emotion_percentages = {emotion: (count / total_records * 100) 
                              for emotion, count in emotion_counts.items()}
        
        # Split recording into segments
        segment_size = max(len(self.emotion_records) // 3, 1)
        beginning = self.emotion_records[:segment_size]
        middle = self.emotion_records[segment_size:2*segment_size]
        end = self.emotion_records[2*segment_size:]
        
        # Get significant emotions in each segment
        beginning_emotions = self._get_significant_emotions(beginning)
        middle_emotions = self._get_significant_emotions(middle)
        end_emotions = self._get_significant_emotions(end)
        
        # Calculate emotional patterns and momentum
        emotion_journey = {
            "beginning": beginning_emotions,
            "middle": middle_emotions,
            "end": end_emotions
        }
        
        # Get significant overall emotions
        significant_emotions = {}
        for emotion, percentage in emotion_percentages.items():
            if percentage >= self.significance_threshold:
                significant_emotions[emotion] = percentage
        
        # Detailed interpretation
        interpretation = self._interpret_emotional_journey(emotion_journey, emotion_percentages)
        
        # Identify emotional shifts and patterns
        significant_shifts = self._identify_emotional_shifts(emotion_journey)
        
        # Educational recommendations
        educational_recommendations = self._generate_educational_recommendations(emotion_journey, significant_emotions)
        
        return {
            "stats": emotion_percentages,
            "duration": total_duration,
            "emotion_journey": emotion_journey,
            "significant_emotions": significant_emotions,
            "interpretation": interpretation,
            "significant_shifts": significant_shifts,
            "educational_recommendations": educational_recommendations
        }
    
    def _get_significant_emotions(self, records):
        if not records:
            return {}
        
        emotion_counts = {}
        for record in records:
            emotion = record['emotion']
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        total = len(records)
        significant = {}
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100
            if percentage >= self.significance_threshold:
                significant[emotion] = percentage
                
        return significant
    
    def _identify_emotional_shifts(self, emotion_journey):
        shifts = []
        
        # Check for emotions that appeared or disappeared
        all_segments = ["beginning", "middle", "end"]
        for i in range(len(all_segments)-1):
            current_segment = all_segments[i]
            next_segment = all_segments[i+1]
            
            current_emotions = set(emotion_journey[current_segment].keys())
            next_emotions = set(emotion_journey[next_segment].keys())
            
            appeared = next_emotions - current_emotions
            disappeared = current_emotions - next_emotions
            
            for emotion in appeared:
                shifts.append(f"{emotion} emerged in the {next_segment} segment")
            
            for emotion in disappeared:
                shifts.append(f"{emotion} diminished after the {current_segment} segment")
                
            # Check for significant increases or decreases
            for emotion in current_emotions.intersection(next_emotions):
                current_pct = emotion_journey[current_segment][emotion]
                next_pct = emotion_journey[next_segment][emotion]
                
                if next_pct >= current_pct * 1.5:
                    shifts.append(f"{emotion} significantly increased from {current_segment} to {next_segment}")
                elif next_pct <= current_pct * 0.6:
                    shifts.append(f"{emotion} significantly decreased from {current_segment} to {next_segment}")
        
        return shifts
    
    def _interpret_emotional_journey(self, emotion_journey, overall_emotions):
        interpretation = "Based on the emotional patterns observed:\n\n"
        
        # Analyze beginning state
        if emotion_journey["beginning"]:
            beginning_emotions = emotion_journey["beginning"]
            interpretation += "At the beginning: "
            
            if "Happy" in beginning_emotions:
                interpretation += "The student started with a positive mindset, showing engagement and readiness to learn. "
            if "Neutral" in beginning_emotions:
                interpretation += "The student began in a calm, receptive state, ready to absorb information. "
                # Add nuance about potential hidden emotions
                interpretation += "Note that neutral expressions may sometimes mask mild confusion or uncertainty. "
            if "Angry" in beginning_emotions:
                interpretation += "The student may have started with frustration or resistance to the subject matter. "
            if "Sad" in beginning_emotions:
                interpretation += "The student showed signs of disengagement or concern at the start. "
            if "Surprise" in beginning_emotions:
                interpretation += "The student showed curiosity or was intrigued by early content. "
            if "Fear" in beginning_emotions:
                interpretation += "The student may have felt anxiety or apprehension about the topic initially. "
        
        # Analyze middle state
        if emotion_journey["middle"]:
            middle_emotions = emotion_journey["middle"]
            interpretation += "\n\nDuring the middle: "
            
            if "Happy" in middle_emotions:
                interpretation += "The student experienced moments of understanding or connection with the material. "
            if "Neutral" in middle_emotions:
                interpretation += "The student maintained steady attention and focus. "
            if "Angry" in middle_emotions:
                interpretation += "The student encountered concepts that were challenging or frustrating. "
            if "Sad" in middle_emotions:
                interpretation += "The student may have struggled with comprehension or engagement. "
            if "Surprise" in middle_emotions:
                interpretation += "The student encountered unexpected concepts that captured their attention. "
            if "Fear" in middle_emotions:
                interpretation += "The student may have felt uncertain about their understanding. "
        
        # Analyze end state
        if emotion_journey["end"]:
            end_emotions = emotion_journey["end"]
            interpretation += "\n\nBy the end: "
            
            if "Happy" in end_emotions:
                interpretation += "The student achieved a sense of accomplishment or satisfaction. "
            if "Neutral" in end_emotions:
                interpretation += "The student maintained composed attention through the conclusion. "
            if "Angry" in end_emotions:
                interpretation += "The student may have been left with unresolved challenges or disagreements with the content. "
            if "Sad" in end_emotions:
                interpretation += "The student may have felt disappointed or unconfident about their learning outcome. "
            if "Surprise" in end_emotions:
                interpretation += "The student experienced new insights or revelations in the final stages. "
            if "Fear" in end_emotions:
                interpretation += "The student may have concerns about applying or remembering the material. "
        
        # Analyze emotional patterns across the session
        interpretation += "\n\nOverall pattern: "
        if "Neutral" in overall_emotions and overall_emotions["Neutral"] > 40:
            interpretation += "High levels of neutral expression dominated the session. This may indicate focused attention, but in educational settings could also represent mild disengagement or difficulty expressing emotions. "
        if "Happy" in overall_emotions and "Neutral" in overall_emotions and overall_emotions["Happy"] > 15:
            interpretation += "The session maintained a generally positive atmosphere with good engagement. "
        if "Sad" in overall_emotions and overall_emotions["Sad"] > 10:
            interpretation += "Periods of sadness or confusion were detected, suggesting potential difficulties with some concepts. "
        if "Angry" in overall_emotions and overall_emotions["Angry"] > 15:
            interpretation += "Frustration was a significant factor throughout the session, possibly indicating challenging content. "
        if "Surprise" in overall_emotions and "Happy" in overall_emotions:
            interpretation += "The learning process included moments of discovery that led to positive engagement. "
        
        return interpretation
    
    def _generate_educational_recommendations(self, emotion_journey, significant_emotions):
        recommendations = []
        
        # Check for persistent negative emotions
        if "Angry" in significant_emotions and significant_emotions["Angry"] > 15:
            if "Angry" in emotion_journey["end"]:
                recommendations.append("Consider revisiting difficult concepts that may have caused frustration.")
            recommendations.append("Break down complex topics into smaller, more digestible segments.")
        
        # Check for engagement level
        if "Neutral" in significant_emotions and significant_emotions["Neutral"] > 40:
            recommendations.append("Incorporate more interactive elements to increase active engagement.")
        
        # Check for positive outcomes
        if "Happy" in emotion_journey["end"]:
            recommendations.append("The positive ending suggests effective learning - build on this success in future sessions.")
        
        # Check for surprise moments
        if "Surprise" in significant_emotions:
            recommendations.append("The moments of surprise indicate effective hook points - consider expanding on these teaching techniques.")
        
        # Check for mixed emotions
        if len(significant_emotions) >= 3:
            recommendations.append("The varied emotional response suggests an engaging but challenging session. Consider providing additional resources for reinforcement.")
        
        # Check for emotional shifts
        beginning_emotions = set(emotion_journey["beginning"].keys())
        end_emotions = set(emotion_journey["end"].keys())
        
        if "Sad" in beginning_emotions and "Happy" in end_emotions:
            recommendations.append("The shift from negative to positive emotion suggests successful scaffolding that helped overcome initial difficulties.")
        
        if "Happy" in beginning_emotions and "Sad" in end_emotions:
            recommendations.append("Consider revisiting the later content that may have caused confusion after initial understanding.")
        
        return recommendations

    def process_frame(self, frame):
        """Process a single frame and detect emotions"""
        try:
            # Make a copy to avoid modifying the original
            img = frame.copy()
            
            # Resize image to speed up processing (50% smaller)
            scale = 0.5
            small_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
            emotions_detected = []
            
            # Use DeepFace with lighter settings
            try:
                results = DeepFace.analyze(
                    small_frame,
                    actions=['emotion'],
                    enforce_detection=False,  # Don't enforce face detection
                    detector_backend='opencv',  # Faster detector
                    silent=True
                )
                
                # Normalize results
                if not isinstance(results, list):
                    results = [results]
                
                for result in results:
                    if 'emotion' in result and 'region' in result:
                        # Get face location
                        x = int(result['region']['x'] / scale)
                        y = int(result['region']['y'] / scale)
                        w = int(result['region']['w'] / scale)
                        h = int(result['region']['h'] / scale)
                        
                        # Extract emotion data
                        emotions_dict = result['emotion']
                        dominant_emotion = result['dominant_emotion']
                        
                        # Format emotion scores as percentages
                        emotion_scores = {k: float(v) for k, v in emotions_dict.items()}
                        
                        emotions_detected.append({
                            "face_location": [x, y, x+w, y+h],
                            "dominant_emotion": dominant_emotion,
                            "emotion_scores": emotion_scores
                        })
                        
                        # Draw rectangle on frame for visualization
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # IMPORTANT: If recording, store the emotion data
                        if self.recording:
                            timestamp = time.time() - self.start_time if self.start_time else 0
                            self.emotion_data.append({
                                "timestamp": timestamp,
                                "emotion": dominant_emotion,
                                "score": emotion_scores.get(dominant_emotion.lower(), 0)
                            })
                            print(f"Recorded emotion: {dominant_emotion} at {timestamp:.2f}s")
            
            except Exception as e:
                print(f"DeepFace error: {e}")
            
            # Store current emotion data for API access
            self.current_emotion_data = emotions_detected
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def calibrate_emotion_sensitivity(self, sad_neutral_ratio=None):
        """
        Calibrate the sensitivity of emotion classification.
        
        Parameters:
        - sad_neutral_ratio: float value between 0.1-1.0 that determines how easily
          Neutral emotions are reclassified as Sad. Lower values make the system more
          sensitive to detecting sadness.
        """
        if sad_neutral_ratio is not None and 0.1 <= sad_neutral_ratio <= 1.0:
            self.sad_neutral_ratio_threshold = sad_neutral_ratio
            print(f"Neutral-to-Sad sensitivity threshold set to: {sad_neutral_ratio}")
            print(f"Lower values will classify more neutral expressions as sad.")
        else:
            print("Invalid sad_neutral_ratio. Please use a value between 0.1 and 1.0")

    def detect_emotions_in_frame(self, frame):
        """Detect faces and emotions in a single frame"""
        try:
            # Get a copy of the frame
            img = frame.copy()
            
            # Convert to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace to detect emotions
            results = DeepFace.analyze(
                rgb_frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # Use OpenCV for more consistent face detection
            )
            
            # Normalize results structure
            if not isinstance(results, list):
                results = [results]
                
            face_emotions = []
            
            for result in results:
                if 'emotion' in result and 'region' in result:
                    # Get face location
                    face_x = result['region']['x']
                    face_y = result['region']['y']
                    face_w = result['region']['w']
                    face_h = result['region']['h']
                    
                    # Ensure coordinates are integers
                    face_x = int(face_x)
                    face_y = int(face_y)
                    face_w = int(face_w)
                    face_h = int(face_h)
                    
                    # Get emotions
                    emotions_dict = result['emotion']
                    dominant_emotion = result['dominant_emotion']
                    
                    face_emotions.append({
                        "face_location": [face_x, face_y, face_x + face_w, face_y + face_h],
                        "dominant_emotion": dominant_emotion,
                        "emotion_scores": emotions_dict
                    })
            
            return face_emotions
        except Exception as e:
            print(f"Error in detect_emotions_in_frame: {str(e)}")
            return []

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