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
        print(f"Recording started at {self.start_time} with unix timestamp")
        return True
    
    def stop_recording(self):
        """Stop the current recording session"""
        if not self.recording:
            print("Warning: stop_recording called but recording was not active")
            return True
        
        self.recording = False
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        
        # Debug output
        record_count = len(self.emotion_records) if hasattr(self, "emotion_records") else 0
        print(f"Recording stopped after {duration:.2f}s with {record_count} emotions recorded")
        
        if record_count > 0:
            print(f"First record: {self.emotion_records[0]}")
            print(f"Last record: {self.emotion_records[-1]}")
            
            # Verify timestamps are correct
            if record_count >= 2:
                start_ts = self.emotion_records[0]['timestamp']
                end_ts = self.emotion_records[-1]['timestamp']
                print(f"First timestamp: {start_ts:.2f}, Last timestamp: {end_ts:.2f}, Duration: {end_ts - start_ts:.2f}s")
        else:
            print("WARNING: No emotion records were collected during recording!")
        
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
            frame_height, frame_width = img.shape[:2]
            
            # Resize image for faster processing
            scale = 0.5
            small_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
            emotions_detected = []
            face_detected = False
            
            try:
                # Process with DeepFace
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
                        # Get face rectangle if available
                        region = result.get('region', {})
                        
                        # Scale coordinates back to original size
                        x = int(region.get('x', 0) / scale)
                        y = int(region.get('y', 0) / scale)
                        w = int(region.get('w', 0) / scale)
                        h = int(region.get('h', 0) / scale)
                        
                        # Apply padding to make box larger (improves visual appearance)
                        padding = int(h * 0.1)  # 10% padding
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame_width - x, w + padding * 2)
                        h = min(frame_height - y, h + padding * 2)
                        
                        # Get emotions
                        emotion_scores = result['emotion']
                        dominant_emotion = result.get('dominant_emotion', 
                                                 max(emotion_scores.items(), key=lambda x: x[1])[0]
                                                ).lower().capitalize()
                        
                        face_detected = True
                        
                        # Update the store with proper coordinates for front-end rendering
                        emotions_detected.append({
                            'face_location': [x, y, x+w, y+h],  # Use list format with correct coordinates
                            'dominant_emotion': dominant_emotion,
                            'emotion_scores': emotion_scores
                        })
                        
                        # Draw rectangle for visualization
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Record emotion data when recording is active
                        if self.recording:
                            timestamp = time.time() - self.start_time if self.start_time else 0
                            emotion_name = dominant_emotion.capitalize()
                            self.emotion_records.append({
                                "timestamp": timestamp,
                                "emotion": emotion_name,
                                "score": emotion_scores.get(dominant_emotion.lower(), 0)
                            })
                
            except Exception as e:
                print(f"DeepFace processing error: {e}")
            
            # Store current emotion data for API access
            self.current_emotion_data = emotions_detected
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
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
        print(f"Recording duration: {duration:.2f}s from {self.start_time} to {self.end_time}")
        
        # Check that we have the start and end times recorded correctly
        if not self.start_time or not self.end_time:
            print("ERROR: Missing start_time or end_time - recording may not have been properly started/stopped")
        
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
                },
                "interpretation": "Insufficient emotion data was captured during the recording session. This may occur if your face was not visible to the camera, lighting was poor, or the emotion detection system had difficulty identifying facial expressions.",
                "educational_recommendations": [
                    "Ensure your face is clearly visible throughout the recording session.",
                    "Check that lighting is adequate, with even illumination on your face.",
                    "Position yourself directly facing the camera for best detection results.",
                    "Avoid covering parts of your face during the recording."
                ]
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
            print(f"Generating analysis with beginning: {list(beginning_percentages.keys())}, middle: {list(middle_percentages.keys())}, end: {list(end_percentages.keys())}")
            
            interpretation = self._generate_interpretation(beginning_percentages, middle_percentages, end_percentages)
            
            recommendations = self._generate_recommendations({
                "beginning": beginning_percentages, 
                "middle": middle_percentages, 
                "end": end_percentages
            }, dominant_emotion)
            
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
        """Generate detailed interpretation of emotional journey with bullet points"""
        try:
            # Get dominant emotions for each phase
            beginning_emotion = max(beginning.items(), key=lambda x: x[1])[0] if beginning else "Neutral"
            middle_emotion = max(middle.items(), key=lambda x: x[1])[0] if middle else "Neutral"
            end_emotion = max(end.items(), key=lambda x: x[1])[0] if end else "Neutral"
            
            interpretation = []
            
            # Beginning phase analysis
            interpretation.append("## Initial Learning Phase")
            
            if beginning_emotion == "Happy":
                interpretation.append("• Students exhibited positive engagement at the beginning, showing readiness to learn")
                interpretation.append("• This suggests successful activation of prior knowledge or effective introductory materials")
            elif beginning_emotion == "Neutral":
                interpretation.append("• Students displayed attentive receptiveness in the initial phase")
                interpretation.append("• This indicates a focused but emotionally reserved approach to the new content")
            elif beginning_emotion == "Sad":
                interpretation.append("• The session began with signs of hesitation or uncertainty")
                interpretation.append("• This possibly reflects anxiety about the subject matter or difficulty with prerequisite knowledge")
            elif beginning_emotion == "Angry":
                interpretation.append("• Initial resistance was detected at the start of the session")
                interpretation.append("• This potentially indicates frustration with previous concepts or a disconnect with expectations")
            elif beginning_emotion == "Surprise":
                interpretation.append("• The session began with cognitive activation through novelty")
                interpretation.append("• This indicates successful capturing of attention through unexpected or intriguing elements")
            
            # Middle phase analysis
            interpretation.append("\n## Core Content Engagement")
            
            if beginning_emotion != middle_emotion:
                interpretation.append(f"• A significant transition from {beginning_emotion} to {middle_emotion} occurred during primary content delivery")
                
                if beginning_emotion in ["Neutral", "Sad"] and middle_emotion == "Happy":
                    interpretation.append("• This positive shift suggests successful cognitive activation as students connected with the material")
                    interpretation.append("• The change indicates effective explanatory approaches that resonated with students")
                elif beginning_emotion in ["Happy", "Neutral"] and middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This downward shift suggests increasing cognitive load possibly exceeding optimal levels")
                    interpretation.append("• Content complexity may have outpaced student comprehension during this phase")
                elif beginning_emotion in ["Sad", "Angry"] and middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• The persistence of challenge indicators suggests continued conceptual barriers")
                    interpretation.append("• Students likely struggled with core content elements throughout this phase")
            else:
                interpretation.append(f"• The {middle_emotion} engagement pattern remained consistent during the primary content delivery")
                
                if middle_emotion == "Happy":
                    interpretation.append("• This suggests appropriate complexity level and successful knowledge construction")
                    interpretation.append("• Students maintained positive cognitive engagement with the material")
                elif middle_emotion == "Neutral":
                    interpretation.append("• This indicates sustained focus but potentially limited affective connection")
                    interpretation.append("• Students were attentive but may benefit from increased emotional investment")
                elif middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This suggests persistent difficulty with core conceptual elements")
                    interpretation.append("• Students continued to experience barriers to comprehension during this phase")
            
            # End phase analysis
            interpretation.append("\n## Knowledge Integration Phase")
            
            if middle_emotion != end_emotion:
                interpretation.append(f"• The concluding segment showed a shift from {middle_emotion} to {end_emotion}")
                
                if middle_emotion in ["Sad", "Angry"] and end_emotion == "Happy":
                    interpretation.append("• This indicates successful conceptual resolution by the end")
                    interpretation.append("• Effective clarification strategies brought clarity to previously challenging concepts")
                elif middle_emotion == "Happy" and end_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This suggests emerging challenges during application or synthesis activities")
                    interpretation.append("• Students may have struggled when asked to independently apply concepts")
                elif middle_emotion in ["Happy", "Neutral"] and end_emotion in ["Neutral"]:
                    interpretation.append("• This reflects a return to baseline cognitive processing as concepts were consolidated")
            else:
                interpretation.append(f"• The {end_emotion} response pattern continued through the conclusion")
                
                if end_emotion == "Happy":
                    interpretation.append("• This suggests successful integration of knowledge and satisfaction with outcomes")
                elif end_emotion == "Neutral":
                    interpretation.append("• This indicates maintained attention but potentially limited transformative impact")
                elif end_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This suggests unresolved conceptual barriers persisted through the conclusion")
            
            # Overall learning trajectory assessment
            interpretation.append("\n## Complete Learning Journey Assessment")
            
            if beginning_emotion == "Neutral" and middle_emotion == "Happy" and end_emotion == "Happy":
                interpretation.append("• This session exhibited an ideal learning progression from receptiveness to engagement")
                interpretation.append("• The pattern indicates effective pedagogical alignment with student cognitive needs")
            elif beginning_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry"]:
                interpretation.append("• The declining trajectory suggests increasing cognitive demands outpaced capacity")
                interpretation.append("• This pattern indicates potential need for additional scaffolding or prerequisite knowledge")
            elif beginning_emotion in ["Sad", "Angry"] and end_emotion in ["Happy", "Neutral"]:
                interpretation.append("• The positive progression demonstrates successful pedagogical intervention")
                interpretation.append("• Teaching strategies effectively addressed initial comprehension barriers")
            elif beginning_emotion == end_emotion:
                interpretation.append(f"• The consistent {beginning_emotion} affect suggests a stable learning experience")
                interpretation.append("• There was limited variability in cognitive-affective engagement throughout")
            else:
                interpretation.append("• The varied emotional pattern suggests a complex learning journey")
                interpretation.append("• Students experienced distinct peaks and valleys in comprehension and engagement")
            
            return "\n".join(interpretation)
        except Exception as e:
            print(f"Error in interpretation generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return "• Insufficient data to provide a detailed emotional journey analysis"

    def _generate_recommendations(self, emotions, dominant_emotion):
        """Generate professional pedagogical recommendations using bullet points"""
        try:
            recommendations = []
            
            # Get emotions from each stage
            beg_emotion = max(emotions["beginning"].items(), key=lambda x: x[1])[0] if emotions.get("beginning") else "Neutral"
            mid_emotion = max(emotions["middle"].items(), key=lambda x: x[1])[0] if emotions.get("middle") else "Neutral"
            end_emotion = max(emotions["end"].items(), key=lambda x: x[1])[0] if emotions.get("end") else "Neutral"
            
            # Primary recommendations based on dominant emotion
            recommendations.append(f"## Pedagogical Recommendations Based on {dominant_emotion} Response Pattern")
            
            if dominant_emotion == "Happy":
                recommendations.append("### Building on Positive Engagement")
                recommendations.append("The strong positive affect suggests successful pedagogical approaches that should be reinforced:")
                recommendations.append("• Leverage the established engagement by introducing more complex analytical tasks")
                recommendations.append("• Implement structured peer teaching components where students articulate understanding")
                recommendations.append("• Connect established concepts to broader theoretical frameworks or applications")
                
            elif dominant_emotion == "Neutral":
                recommendations.append("### Enhancing Cognitive-Affective Connection")
                recommendations.append("While attention was maintained, deeper engagement could be fostered through:")
                recommendations.append("• Incorporate discipline-specific cases that create emotional anchors for abstract concepts")
                recommendations.append("• Introduce conceptual conflicts that require students to evaluate competing positions")
                recommendations.append("• Present key concepts through multiple sensory modalities for increased engagement")
                
            elif dominant_emotion == "Sad":
                recommendations.append("### Addressing Comprehension Barriers")
                recommendations.append("The affective pattern suggests potential conceptual obstacles that could be addressed through:")
                recommendations.append("• Implement brief diagnostic activities to identify specific prerequisite knowledge deficiencies")
                recommendations.append("• Provide intermediate representations that bridge familiar knowledge to new concepts")
                recommendations.append("• Structure content with more frequent achievement benchmarks to build confidence")
                
            elif dominant_emotion == "Angry":
                recommendations.append("### Resolving Cognitive Friction")
                recommendations.append("The frustration indicators suggest needed adjustments in these areas:")
                recommendations.append("• Segment complex material into smaller conceptual units with explicit integration pathways")
                recommendations.append("• Provide clearer algorithmic approaches to complex tasks with graduated scaffolding")
                recommendations.append("• Demonstrate productive approaches through transparent expert thinking processes")
                
            elif dominant_emotion == "Surprise":
                recommendations.append("### Capitalizing on Cognitive Activation")
                recommendations.append("The surprise response indicates successful cognitive activation that can be leveraged:")
                recommendations.append("• Continue highlighting unexpected relationships or counterintuitive outcomes")
                recommendations.append("• Structure discovery-based activities that build on initial curiosity")
                recommendations.append("• Explicitly address the gap between intuition and disciplinary understanding")
            
            # Add specific recommendations based on emotional journey
            recommendations.append("\n### Learning Progression Optimization")
            
            # Specific pattern-based recommendations
            if beg_emotion in ["Sad", "Angry"] and end_emotion in ["Sad", "Angry"]:
                recommendations.append("The persistent challenge indicators suggest need for fundamental adjustments:")
                recommendations.append("• Create targeted resources addressing foundational knowledge gaps")
                recommendations.append("• Provide explicit visual mapping of how concepts interconnect")
                recommendations.append("• Restructure content progression with more gradual complexity increases")
                
            elif beg_emotion in ["Neutral", "Sad"] and end_emotion == "Happy":
                recommendations.append("Your session achieved effective progression toward understanding:")
                recommendations.append("• Document the specific explanatory approaches that facilitated the positive shift")
                recommendations.append("• Establish stronger initial concrete examples before abstract principles")
                recommendations.append("• Gradually increase complexity as confidence develops")
                
            elif beg_emotion == "Happy" and end_emotion in ["Sad", "Angry"]:
                recommendations.append("The declining engagement pattern suggests attention to:")
                recommendations.append("• Redistribute complex tasks more evenly throughout the session")
                recommendations.append("• Implement brief application interludes before introducing new complexity")
                recommendations.append("• Provide more explicit connections between sequential concepts")
            
            # Teaching approach recommendations
            recommendations.append("\n### Instructional Method Refinement")
            
            if "Happy" in emotions.get("beginning", {}) and emotions["beginning"].get("Happy", 0) > 40:
                recommendations.append("The strong initial engagement suggests effective activation strategies:")
                recommendations.append("• Document and systematize your successful approach to beginning new topics")
                recommendations.append("• Expand the initial engagement techniques with advance organizers")
                
            if "Happy" in emotions.get("middle", {}) and emotions["middle"].get("Happy", 0) > 50:
                recommendations.append("Your core content delivery methods demonstrate strong effectiveness:")
                recommendations.append("• Identify and formalize the explanation patterns that generated positive response")
                recommendations.append("• Leverage this successful approach through structured knowledge sharing")
                
            if any(emotions.get(phase, {}).get("Sad", 0) > 30 for phase in ["beginning", "middle", "end"]):
                recommendations.append("Address comprehension barriers through targeted methodological adjustments:")
                recommendations.append("• Present challenging concepts through alternative representational forms")
                recommendations.append("• Explicitly model expert thinking processes when approaching difficult concepts")
                
            return recommendations
        except Exception as e:
            print(f"Error in recommendation generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return ["• Consider reviewing the session materials based on student emotion patterns",
                    "• Use multiple teaching modalities to address diverse learning needs",
                    "• Break complex topics into smaller segments to improve comprehension"]

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
            print("Error: Can't receive frame (stream end?). Exiting...")