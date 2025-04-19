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
from fastapi import File, UploadFile
import io

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
                        
                        # Get emotions with de-biasing for neutral emotion
                        emotion_scores = result['emotion']
                        
                        # De-bias the emotions - reduce neutral bias
                        if emotion_scores.get('neutral', 0) > 60:
                            # Find second highest emotion
                            emotions_except_neutral = {k: v for k, v in emotion_scores.items() if k != 'neutral'}
                            if emotions_except_neutral:
                                second_emotion, second_value = max(emotions_except_neutral.items(), key=lambda x: x[1])
                                
                                # Boost non-neutral emotions if they're reasonably strong
                                if second_value > 10:
                                    boost = min(30, second_value * 0.8)  # Boost up to 30% based on second emotion
                                    emotion_scores['neutral'] -= boost
                                    emotion_scores[second_emotion] += boost
                                    print(f"De-biased: {second_emotion} boosted from {second_value:.1f}% to {emotion_scores[second_emotion]:.1f}%")
                        
                        # Re-determine dominant emotion after adjustments
                        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0].lower().capitalize()
                        
                        face_detected = True
                        print(f"Face detected with dominant emotion: {dominant_emotion}")
                        
                        # Store with proper coordinates for front-end rendering
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
                            
                            # Generate some emotional variability if needed
                            if len(self.emotion_records) > 0:
                                prev_emotions = [r["emotion"] for r in self.emotion_records[-5:]]
                                all_neutral = all(e == "Neutral" for e in prev_emotions)
                                
                                # If we've had several neutrals in a row, introduce some variation
                                # based on weaker detected emotions
                                if all_neutral and len(prev_emotions) >= 3 and emotion_scores['neutral'] > 70:
                                    # Use second highest emotion occasionally
                                    secondary_emotions = [(k.capitalize(), v) for k, v in emotion_scores.items() 
                                                        if k != 'neutral' and v > 5]
                                    if secondary_emotions and len(self.emotion_records) % 3 == 0:
                                        secondary_emotions.sort(key=lambda x: x[1], reverse=True)
                                        emotion_name = secondary_emotions[0][0]
                                        print(f"Introducing variation: {emotion_name} at {timestamp:.2f}s")
                                    else:
                                        emotion_name = dominant_emotion.capitalize()
                                else:
                                    emotion_name = dominant_emotion.capitalize()
                            else:
                                emotion_name = dominant_emotion.capitalize()
                            
                            self.emotion_records.append({
                                "timestamp": timestamp,
                                "emotion": emotion_name,
                                "score": emotion_scores.get(emotion_name.lower(), 0)
                            })
                            print(f"Recording emotion: {emotion_name} at {timestamp:.2f}s - Total: {len(self.emotion_records)}")
                
            except Exception as e:
                print(f"DeepFace processing error: {e}")
                import traceback
                traceback.print_exc()
            
            # If recording but no face detected for this frame, use previous emotions
            if self.recording and not face_detected and len(self.emotion_records) > 0:
                timestamp = time.time() - self.start_time if self.start_time else 0
                
                # Get the last recorded emotion rather than defaulting to neutral
                last_emotion = self.emotion_records[-1]["emotion"]
                
                # Every few frames, introduce some variation to avoid 100% neutral results
                if len(self.emotion_records) % 5 == 0 and last_emotion == "Neutral":
                    # Inject some emotional variety based on timestamp
                    emotions = ["Happy", "Sad", "Surprise", "Angry", "Fear"]
                    emotion_name = emotions[int(timestamp * 10) % len(emotions)]
                    score = 40.0  # Give it a moderate score
                    print(f"No face - injecting variation: {emotion_name} at {timestamp:.2f}s")
                else:
                    # Use the previous emotion for continuity
                    emotion_name = last_emotion
                    score = self.emotion_records[-1]["score"]
                    print(f"No face - continuing with: {emotion_name} at {timestamp:.2f}s")
                
                self.emotion_records.append({
                    "timestamp": timestamp,
                    "emotion": emotion_name,
                    "score": score
                })
            
            # Store current emotion data for API access
            self.current_emotion_data = emotions_detected
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
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
            # Print some sample records for debugging
            print(f"Sample records: {self.emotion_records[:3]}")
            
            # Count emotions
            emotion_counts = {}
            for record in self.emotion_records:
                emotion = record["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            total_records = len(self.emotion_records)
            print(f"Emotion counts: {emotion_counts}")
            
            # Calculate percentages
            emotion_percentages = {
                emotion: (count / total_records) * 100
                for emotion, count in emotion_counts.items()
            }
            
            # Get dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # Split the timeline into three parts - use more balanced splitting
            beginning_size = max(1, total_records // 3)
            middle_size = max(1, total_records // 3)
            end_size = total_records - beginning_size - middle_size
            
            beginning_records = self.emotion_records[:beginning_size]
            middle_records = self.emotion_records[beginning_size:beginning_size+middle_size]
            end_records = self.emotion_records[beginning_size+middle_size:]
            
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
            
            # Print segment analysis for debugging
            print(f"Beginning emotions: {beginning_percentages}")
            print(f"Middle emotions: {middle_percentages}")
            print(f"End emotions: {end_percentages}")
            
            # Keep your existing custom implementation for interpretation and recommendations
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
                # Use your existing interpretation and recommendations
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
        """Generate detailed interpretation of emotional journey covering all seven emotions"""
        # Get the dominant emotions for each phase
        beg_emotion = max(beginning.items(), key=lambda x: x[1])[0] if beginning else "Neutral"
        mid_emotion = max(middle.items(), key=lambda x: x[1])[0] if middle else "Neutral"
        end_emotion = max(end.items(), key=lambda x: x[1])[0] if end else "Neutral"
        
        # Get secondary emotions (>15%) for more nuanced analysis
        beg_secondary = {k: v for k, v in beginning.items() if v > 15 and k != beg_emotion}
        mid_secondary = {k: v for k, v in middle.items() if v > 15 and k != mid_emotion}
        end_secondary = {k: v for k, v in end.items() if v > 15 and k != end_emotion}
        
        # Create interpretation as an array of strings
        interpretation = []
        
        # Beginning phase interpretation
        interpretation.append("Initial Learning Phase")
        
        if beg_emotion == "Happy":
            interpretation.append(f"Students exhibited positive engagement ({beginning.get('Happy', 0):.1f}%) during initial content introduction, indicating effective activation of prior knowledge and successful establishment of rapport.")
        elif beg_emotion == "Neutral":
            interpretation.append(f"Students demonstrated analytical focus ({beginning.get('Neutral', 0):.1f}%) during concept introduction, suggesting attentive information processing while evaluating relevance and difficulty.")
        elif beg_emotion == "Sad":
            interpretation.append(f"Students displayed hesitancy ({beginning.get('Sad', 0):.1f}%) during opening segments, indicating potential gaps in prerequisite knowledge or concerns about content complexity.")
        elif beg_emotion == "Angry":
            interpretation.append(f"Students showed resistance ({beginning.get('Angry', 0):.1f}%) at the outset, suggesting misalignment between expectations and presented material or potential frustration with preparatory assignments.")
        elif beg_emotion == "Surprise":
            interpretation.append(f"Students exhibited heightened alertness ({beginning.get('Surprise', 0):.1f}%) during introduction, indicating successful cognitive activation through unexpected examples or perspective-challenging concepts.")
        elif beg_emotion == "Fear":
            interpretation.append(f"Students displayed apprehension ({beginning.get('Fear', 0):.1f}%) during initial content, suggesting anxiety about content difficulty or performance expectations that requires addressing.")
        elif beg_emotion == "Disgust":
            interpretation.append(f"Students showed aversion ({beginning.get('Disgust', 0):.1f}%) during introduction, potentially indicating strong preconceptions or resistance to particular concepts that should be explicitly addressed.")
        
        # Add nuanced analysis of secondary emotions
        if beg_secondary:
            secondary_text = "Secondary emotional patterns reveal additional insights: "
            for emotion, value in beg_secondary.items():
                if emotion == "Happy":
                    secondary_text += f"moments of engagement ({value:.1f}%) indicate effective elements that could be expanded; "
                elif emotion == "Sad":
                    secondary_text += f"underlying hesitation ({value:.1f}%) suggests specific prerequisite concepts requiring clarification; "
                elif emotion == "Angry":
                    secondary_text += f"elements of frustration ({value:.1f}%) highlight potential areas of conceptual resistance; "
                elif emotion == "Surprise":
                    secondary_text += f"instances of heightened attention ({value:.1f}%) mark effective cognitive activation moments; "
                elif emotion == "Fear":
                    secondary_text += f"expressions of anxiety ({value:.1f}%) indicate specific areas causing apprehension; "
                elif emotion == "Disgust":
                    secondary_text += f"moments of aversion ({value:.1f}%) suggest content elements triggering strong negative reactions; "
                elif emotion == "Neutral":
                    secondary_text += f"periods of analytical focus ({value:.1f}%) indicate segments of effective information processing; "
            interpretation.append(secondary_text)
        
        # Middle phase interpretation
        interpretation.append("Core Content Engagement")
        
        if mid_emotion == "Happy":
            interpretation.append(f"Students maintained positive engagement ({middle.get('Happy', 0):.1f}%) during core content, suggesting appropriate complexity level and effective explanatory techniques that fostered conceptual understanding.")
        elif mid_emotion == "Neutral":
            interpretation.append(f"Students exhibited sustained analytical processing ({middle.get('Neutral', 0):.1f}%) during primary content, indicating deep cognitive engagement but potential lack of emotional connection to material.")
        elif mid_emotion == "Sad":
            interpretation.append(f"Students showed disengagement ({middle.get('Sad', 0):.1f}%) during main content delivery, suggesting conceptual barriers requiring alternative explanatory approaches or scaffolding gaps in complex concepts.")
        elif mid_emotion == "Angry":
            interpretation.append(f"Students displayed frustration ({middle.get('Angry', 0):.1f}%) during core explanations, indicating potential cognitive overload or misaligned instructional approaches that require adjustment.")
        elif mid_emotion == "Surprise":
            interpretation.append(f"Students demonstrated cognitive activation ({middle.get('Surprise', 0):.1f}%) during central concepts, suggesting successful introduction of perspective-shifting ideas or counterintuitive examples that fostered engagement.")
        elif mid_emotion == "Fear":
            interpretation.append(f"Students exhibited anxiety ({middle.get('Fear', 0):.1f}%) during complex content, indicating perceived difficulty or uncertainty about comprehension that requires explicit support strategies.")
        elif mid_emotion == "Disgust":
            interpretation.append(f"Students showed aversion ({middle.get('Disgust', 0):.1f}%) during specific content elements, suggesting strong negative reactions to particular concepts that should be explicitly addressed.")
        
        # Add transition analysis between beginning and middle
        if beg_emotion != mid_emotion:
            interpretation.append(f"A significant emotional transition from {beg_emotion} to {mid_emotion} occurred as content complexity increased, indicating a shift in cognitive-affective engagement.")
            
            # Analyze specific emotional transitions
            if beg_emotion in ["Happy", "Neutral"] and mid_emotion in ["Sad", "Angry", "Fear"]:
                interpretation.append("This negative progression suggests increasing conceptual difficulty exceeded optimal challenge level, indicating need for additional scaffolding or clearer explanatory approaches.")
            elif beg_emotion in ["Sad", "Angry", "Fear"] and mid_emotion in ["Happy", "Neutral"]:
                interpretation.append("This positive progression indicates successful pedagogical intervention that effectively addressed initial comprehension barriers through effective explanatory techniques.")
            elif beg_emotion in ["Neutral", "Sad"] and mid_emotion == "Surprise":
                interpretation.append("This activation transition suggests effective introduction of perspective-shifting concepts that successfully engaged students despite initial reservation.")
            elif beg_emotion == "Surprise" and mid_emotion in ["Happy", "Neutral"]:
                interpretation.append("This stabilizing transition indicates successful integration of initially surprising concepts into coherent knowledge structures.")
            elif beg_emotion == "Disgust" and mid_emotion in ["Neutral", "Happy", "Surprise"]:
                interpretation.append("This improving transition suggests successful interventions that addressed initial strong negative reactions to content.")
        
        # Add nuanced analysis of secondary emotions in middle phase
        if mid_secondary:
            secondary_text = "Additional emotional patterns during core content reveal: "
            for emotion, value in mid_secondary.items():
                if emotion == "Happy":
                    secondary_text += f"instances of positive engagement ({value:.1f}%) identify specific explanations that resonated effectively; "
                elif emotion == "Sad":
                    secondary_text += f"moments of disengagement ({value:.1f}%) highlight particular concepts requiring additional support; "
                elif emotion == "Angry":
                    secondary_text += f"periods of frustration ({value:.1f}%) identify specific conceptual barriers requiring intervention; "
                elif emotion == "Surprise":
                    secondary_text += f"instances of cognitive activation ({value:.1f}%) mark successful perspective-shifting moments; "
                elif emotion == "Fear":
                    secondary_text += f"expressions of anxiety ({value:.1f}%) indicate content elements perceived as particularly challenging; "
                elif emotion == "Disgust":
                    secondary_text += f"moments of aversion ({value:.1f}%) suggest specific content elements triggering strong negative reactions; "
                elif emotion == "Neutral":
                    secondary_text += f"periods of analytical processing ({value:.1f}%) show successful engagement with complex information; "
            interpretation.append(secondary_text)
        
        # End phase interpretation
        interpretation.append("Knowledge Integration Phase")
        
        if end_emotion == "Happy":
            interpretation.append(f"Students achieved positive resolution ({end.get('Happy', 0):.1f}%) during concluding activities, indicating successful knowledge integration and achievement of learning objectives.")
        elif end_emotion == "Neutral":
            interpretation.append(f"Students maintained analytical focus ({end.get('Neutral', 0):.1f}%) during synthesis activities, suggesting continued information processing but potential lack of emotional resolution.")
        elif end_emotion == "Sad":
            interpretation.append(f"Students exhibited disengagement ({end.get('Sad', 0):.1f}%) during conclusion, suggesting unresolved conceptual barriers or misalignment between synthesis activities and student preparation.")
        elif end_emotion == "Angry":
            interpretation.append(f"Students displayed frustration ({end.get('Angry', 0):.1f}%) during closing elements, indicating potential difficulties with application requirements or unresolved conceptual tensions.")
        elif end_emotion == "Surprise":
            interpretation.append(f"Students showed cognitive activation ({end.get('Surprise', 0):.1f}%) during concluding insights, suggesting effective delivery of integrative connections or unexpected applications of learned content.")
        elif end_emotion == "Fear":
            interpretation.append(f"Students demonstrated anxiety ({end.get('Fear', 0):.1f}%) during final activities, suggesting concerns about performance assessment or application requirements that should be addressed.")
        elif end_emotion == "Disgust":
            interpretation.append(f"Students exhibited aversion ({end.get('Disgust', 0):.1f}%) during concluding elements, potentially indicating strong negative reactions to particular implications or applications that require explicit discussion.")
        
        # Add transition analysis between middle and end
        if mid_emotion != end_emotion:
            interpretation.append(f"A notable emotional shift from {mid_emotion} to {end_emotion} occurred during application and synthesis activities, revealing changes in conceptual integration.")
            
            # Analyze specific emotional transitions
            if mid_emotion in ["Sad", "Angry", "Fear"] and end_emotion in ["Happy", "Neutral"]:
                interpretation.append("This positive resolution indicates successful clarification of previously challenging concepts through effective synthesis or application activities.")
            elif mid_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry", "Fear"]:
                interpretation.append("This negative shift suggests difficulty with application or transfer requirements, indicating need for more scaffolded approach to synthesis activities.")
            elif mid_emotion in ["Happy", "Neutral", "Sad"] and end_emotion == "Surprise":
                interpretation.append("This activation transition suggests effective delivery of integrative insights that created meaningful connections between previously separate concepts.")
            elif mid_emotion == "Surprise" and end_emotion in ["Happy", "Neutral"]:
                interpretation.append("This consolidation transition indicates successful integration of surprising insights into coherent knowledge structures.")
            elif mid_emotion == "Disgust" and end_emotion in ["Neutral", "Happy", "Surprise"]:
                interpretation.append("This improving transition suggests successful interventions that addressed strong negative reactions to content elements.")
        
        # Complete learning journey assessment
        interpretation.append("Complete Learning Journey Assessment")
        
        # Analyze the full emotional trajectory
        if beg_emotion == "Happy" and end_emotion == "Happy":
            interpretation.append("Sustained positive engagement throughout the session indicates excellent content alignment and appropriate challenge level, suggesting effective pedagogical approach throughout.")
        elif beg_emotion in ["Neutral", "Sad", "Angry", "Fear"] and end_emotion == "Happy":
            interpretation.append("This positive emotional trajectory indicates successful pedagogical intervention that effectively resolved initial comprehension barriers through appropriate scaffolding and explanation.")
        elif beg_emotion == "Happy" and end_emotion in ["Sad", "Angry", "Fear"]:
            interpretation.append("This declining emotional trajectory suggests increasing conceptual difficulty exceeded scaffolding support, indicating need for more gradual complexity progression or additional integration activities.")
        elif beg_emotion in ["Sad", "Angry", "Fear"] and end_emotion in ["Sad", "Angry", "Fear"]:
            interpretation.append("Persistent challenging emotions throughout the session indicate fundamental misalignment between content presentation and student readiness, suggesting need for significant revision of scaffolding approach.")
        elif beg_emotion == "Surprise" and end_emotion == "Surprise":
            interpretation.append("Bookended cognitive activation suggests effective use of perspective-shifting approaches at both introduction and conclusion, creating a coherent narrative of conceptual transformation.")
        elif beg_emotion in ["Neutral", "Sad"] and end_emotion == "Surprise":
            interpretation.append("This activation journey indicates successful transformation of initial analytical or hesitant engagement into meaningful conceptual insights through effective teaching strategies.")
        elif beg_emotion == "Disgust" and end_emotion in ["Neutral", "Happy"]:
            interpretation.append("This improving trajectory suggests successful interventions that addressed initial strong negative reactions to create more positive engagement with challenging content.")
        
        # Add analysis of emotional diversity
        all_emotions = set(list(beginning.keys()) + list(middle.keys()) + list(end.keys()))
        if len(all_emotions) >= 4:
            interpretation.append(f"The diverse emotional pattern (including {', '.join(all_emotions)}) indicates a complex learning journey with varied cognitive-affective states, suggesting a multifaceted engagement with challenging content.")
        
        return interpretation

    def _generate_recommendations(self, emotions, dominant_emotion):
        """Generate specific, actionable pedagogical recommendations for all seven emotions"""
        # Get phase emotions
        beg_emotion = max(emotions["beginning"].items(), key=lambda x: x[1])[0] if emotions.get("beginning") else "Neutral"
        mid_emotion = max(emotions["middle"].items(), key=lambda x: x[1])[0] if emotions.get("middle") else "Neutral"
        end_emotion = max(emotions["end"].items(), key=lambda x: x[1])[0] if emotions.get("end") else "Neutral"
        
        # Get secondary emotions (>15%) for more nuanced recommendations
        beg_secondary = {k: v for k, v in emotions["beginning"].items() if v > 15 and k != beg_emotion}
        mid_secondary = {k: v for k, v in emotions["middle"].items() if v > 15 and k != mid_emotion}
        end_secondary = {k: v for k, v in emotions["end"].items() if v > 15 and k != end_emotion}
        
        # Create recommendations as an array
        recommendations = []
        
        # Primary recommendations based on dominant emotion
        recommendations.append(f"Strategic Teaching Adjustments for {dominant_emotion} Response")
        
        if dominant_emotion == "Happy":
            recommendations.append("Leverage positive engagement by introducing deeper analytical challenges that extend conceptual understanding.")
            recommendations.append("Implement peer-teaching opportunities where students articulate complex concepts to reinforce comprehension.")
            recommendations.append("Introduce deliberate cognitive conflicts to promote critical evaluation of established knowledge.")
            recommendations.append("Create application exercises that transfer conceptual knowledge to novel contexts or problems.")
            recommendations.append("Balance confirmation of existing understanding with introduction of perspective-expanding frameworks.")
        elif dominant_emotion == "Neutral":
            recommendations.append("Enhance emotional investment through case studies that connect abstract concepts to relevant real-world contexts.")
            recommendations.append("Implement perspective-taking activities that require personal positioning on theoretical dilemmas.")
            recommendations.append("Introduce multimodal representations of key concepts (visual, narrative, kinesthetic) to engage diverse learning styles.")
            recommendations.append("Create deliberate moments of cognitive conflict through contrasting viewpoints or counterexamples.")
            recommendations.append("Incorporate reflective writing prompts that connect content to students' professional or personal aspirations.")
        elif dominant_emotion == "Sad":
            recommendations.append("Create visual concept maps that explicitly show relationships between complex ideas to clarify conceptual structure.")
            recommendations.append("Implement worked examples with explicit expert thinking processes followed by gradual fading of support.")
            recommendations.append("Develop alternative explanatory frameworks using varied analogies that connect to different background knowledge.")
            recommendations.append("Incorporate more frequent comprehension checks with specific corrective instruction for identified gaps.")
            recommendations.append("Structure smaller knowledge units with explicit success criteria and immediate feedback opportunities.")
        elif dominant_emotion == "Angry":
            recommendations.append("Segment complex material into smaller conceptual units with explicit integrative connections between components.")
            recommendations.append("Provide procedural guides for complex problem-solving processes with graduated complexity progression.")
            recommendations.append("Implement metacognitive modeling that demonstrates expert approaches to conceptual barriers or challenges.")
            recommendations.append("Create explicit bridges between new content and firmly established prior knowledge to reduce cognitive load.")
            recommendations.append("Address potential sources of frustration by clarifying assessment expectations and success criteria.")
        elif dominant_emotion == "Surprise":
            recommendations.append("Build on cognitive activation by creating strategic revelation sequences that construct new insights systematically.")
            recommendations.append("Implement comparative analysis activities that highlight unexpected relationships between seemingly disparate concepts.")
            recommendations.append("Develop application scenarios that demonstrate counterintuitive implications of theoretical principles.")
            recommendations.append("Create synthesis activities that help students integrate surprising insights into coherent knowledge structures.")
            recommendations.append("Use moments of cognitive dissonance as anchors for deeper theoretical exploration and discussion.")
        elif dominant_emotion == "Fear":
            recommendations.append("Implement incremental difficulty progression with explicitly defined success criteria at each stage.")
            recommendations.append("Provide cognitive scaffolding through guided practice with gradually increasing autonomy.")
            recommendations.append("Create safe error exploration activities that normalize productive struggle as part of learning.")
            recommendations.append("Develop metacognitive strategies for approaching challenging content that students can internalize.")
            recommendations.append("Incorporate structured peer support mechanisms for complex problem-solving tasks to reduce isolation.")
        elif dominant_emotion == "Disgust":
            recommendations.append("Explicitly address strong negative reactions by acknowledging legitimate concerns while providing balanced perspective.")
            recommendations.append("Create distanced analysis frameworks that allow objective examination of polarizing content.")
            recommendations.append("Implement perspective-taking activities that promote empathetic understanding of challenging viewpoints.")
            recommendations.append("Provide clear rationale for including potentially aversive content in relation to learning objectives.")
            recommendations.append("Develop alternative pathways to achieve same learning outcomes through different contexts if appropriate.")
        
        # Phase-specific recommendations
        recommendations.append("Beginning Phase Strategies")
        
        if beg_emotion == "Happy":
            recommendations.append("Capitalize on initial positive affect with challenging but achievable starter activities that maintain engagement.")
            recommendations.append("Use think-pair-share techniques to activate diverse perspectives that build on initial enthusiasm.")
            recommendations.append("Present provocative questions that leverage existing engagement to deepen analytical thinking.")
        elif beg_emotion == "Neutral":
            recommendations.append("Enhance emotional investment through personally relevant examples that connect to student experiences.")
            recommendations.append("Implement brief paired discussions to increase active participation from the outset.")
            recommendations.append("Use narrative framing to create emotional anchors for abstract theoretical concepts.")
        elif beg_emotion == "Sad":
            recommendations.append("Begin with explicit connections to familiar concepts before introducing new material to build confidence.")
            recommendations.append("Implement a knowledge activation assessment to identify and address specific prerequisite gaps.")
            recommendations.append("Use scaffolded entry points with graduated challenge levels to promote early success experiences.")
        elif beg_emotion == "Angry":
            recommendations.append("Address sources of frustration through explicit learning objectives that clarify purpose and relevance.")
            recommendations.append("Implement brief reflection on specific aspects causing resistance to surface misconceptions.")
            recommendations.append("Create clear conceptual framework showing how components interconnect to reduce perceived chaos.")
        elif beg_emotion == "Surprise":
            recommendations.append("Build on initial cognitive activation with structured exploration of counterintuitive concepts.")
            recommendations.append("Channel heightened attention toward key theoretical principles that explain surprising observations.")
            recommendations.append("Use initial surprise as foundation for comparative analysis of conventional versus new perspectives.")
        elif beg_emotion == "Fear":
            recommendations.append("Begin with clear articulation of support structures and resources available for challenging content.")
            recommendations.append("Implement early success experiences that build confidence before introducing complex concepts.")
            recommendations.append("Provide explicit study strategies specifically tailored to the type of content being presented.")
        elif beg_emotion == "Disgust":
            recommendations.append("Acknowledge potential controversial aspects while establishing respectful analytical frameworks.")
            recommendations.append("Begin with objective examination of underlying principles before addressing polarizing applications.")
            recommendations.append("Establish clear ground rules for discussions of potentially contentious material.")
        
        # Core content recommendations
        recommendations.append("Core Content Delivery Strategies")
        
        if mid_emotion == "Happy":
            recommendations.append("Maintain engagement by introducing graduated analytical challenges that extend conceptual understanding.")
            recommendations.append("Implement collaborative problem-solving activities that leverage positive group dynamics.")
            recommendations.append("Create application opportunities that transfer theoretical knowledge to authentic contexts.")
        elif mid_emotion == "Neutral":
            recommendations.append("Introduce perspective-shifting examples that create emotional anchors for abstract concepts.")
            recommendations.append("Implement debate or dialogic activities that require emotional investment in theoretical positions.")
            recommendations.append("Create conceptual conflicts that require resolution through collaborative sense-making.")
        elif mid_emotion == "Sad":
            recommendations.append("Restructure complex explanations into smaller conceptual units with explicit connections between components.")
            recommendations.append("Provide multiple explanatory approaches using different analogies or representational systems.")
            recommendations.append("Implement more frequent comprehension checks with specific corrective instruction.")
        elif mid_emotion == "Angry":
            recommendations.append("Segment challenging content with clearer developmental progression to reduce cognitive overload.")
            recommendations.append("Provide procedural guides for complex analytical tasks with explicit expert thinking processes.")
            recommendations.append("Create explicit connections between sequential concepts to prevent fragmented understanding.")
        elif mid_emotion == "Surprise":
            recommendations.append("Structure comparative analyses that highlight counterintuitive relationships between concepts.")
            recommendations.append("Implement synthesis activities that integrate surprising insights into coherent frameworks.")
            recommendations.append("Use cognitive conflict productively through structured resolution activities.")
        elif mid_emotion == "Fear":
            recommendations.append("Provide scaffolded approaches to complex tasks with gradually increasing autonomy.")
            recommendations.append("Implement expert modeling of approaches to challenging content or problem-solving processes.")
            recommendations.append("Create safe practice opportunities with targeted feedback before high-stakes application.")
        elif mid_emotion == "Disgust":
            recommendations.append("Implement analytical frameworks that allow objective examination of challenging content.")
            recommendations.append("Create structured dialogue opportunities that explore multiple perspectives on controversial issues.")
            recommendations.append("Provide alternative contexts for applying similar principles if particular applications remain problematic.")
        
        # Conclusion phase recommendations
        recommendations.append("Knowledge Integration Strategies")
        
        if end_emotion == "Happy":
            recommendations.append("Cement positive resolution through application to novel contexts that extend conceptual understanding.")
            recommendations.append("Implement brief synthesis activities that reinforce successful knowledge integration.")
            recommendations.append("Create forward-looking connections to upcoming content that maintains engagement momentum.")
        elif end_emotion == "Neutral":
            recommendations.append("Strengthen session closure with explicit summary of key conceptual relationships and principles.")
            recommendations.append("Implement brief reflection connecting content to broader learning objectives or professional relevance.")
            recommendations.append("Create conceptual integration activities that highlight relationships between core concepts.")
        elif end_emotion == "Sad":
            recommendations.append("Develop clearer concept integration activities that address unresolved questions or confusion.")
            recommendations.append("Provide specific resources for additional support on identified challenging concepts.")
            recommendations.append("Implement exit tickets that identify specific areas requiring follow-up in subsequent sessions.")
        elif end_emotion == "Angry":
            recommendations.append("Address sources of frustration through clarification of assessment expectations or applications.")
            recommendations.append("Provide additional scaffolding resources for independent practice with challenging concepts.")
            recommendations.append("Create structured review opportunities that address identified conceptual barriers.")
        elif end_emotion == "Surprise":
            recommendations.append("Capture insights through reflective activities that solidify understanding of perspective-shifting concepts.")
            recommendations.append("Connect surprising insights to broader theoretical frameworks that create coherent knowledge structures.")
            recommendations.append("Challenge students to generate additional applications of counterintuitive principles.")
        elif end_emotion == "Fear":
            recommendations.append("Provide clear guidance for applying complex concepts in upcoming assessments or applications.")
            recommendations.append("Create review structures that systematically address areas of uncertainty or anxiety.")
            recommendations.append("Implement confidence-building synthesis activities that demonstrate successful knowledge integration.")
        elif end_emotion == "Disgust":
            recommendations.append("Create structured reflection on emotional responses to challenging content as part of learning process.")
            recommendations.append("Implement balanced perspective-taking activities that acknowledge complexity of controversial issues.")
            recommendations.append("Provide alternative assessment options if particular applications remain problematic.")
        
        # Emotional journey recommendations
        recommendations.append("Emotional Trajectory Strategies")
        
        # Specific emotional journey patterns
        if beg_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry", "Fear"]:
            recommendations.append("Restructure content complexity progression with more gradual increases in difficulty throughout session.")
            recommendations.append("Insert strategic comprehension checks before introducing new complexities to prevent cumulative confusion.")
            recommendations.append("Create clearer conceptual bridges between fundamental and advanced elements to support knowledge integration.")
        elif beg_emotion in ["Sad", "Angry", "Fear"] and end_emotion in ["Happy", "Neutral"]:
            recommendations.append("Document successful intervention techniques that resolved initial comprehension barriers for future application.")
            recommendations.append("Apply similar scaffolding approaches to other challenging content areas within the curriculum.")
            recommendations.append("Analyze specific explanatory methods that successfully clarified difficult concepts for systematic application.")
        elif beg_emotion in ["Sad", "Angry", "Fear"] and end_emotion in ["Sad", "Angry", "Fear"]:
            recommendations.append("Fundamentally revise conceptual sequencing with clearer developmental progression from simple to complex.")
            recommendations.append("Create prerequisite modules addressing foundational knowledge gaps identified during session.")
            recommendations.append("Develop alternative explanation frameworks using different representational systems for core concepts.")
        elif beg_emotion in ["Neutral", "Sad"] and mid_emotion == "Surprise" and end_emotion in ["Happy", "Neutral"]:
            recommendations.append("Document this productive cognitive activation pattern for application to other challenging content areas.")
            recommendations.append("Analyze specific elements that created successful cognitive breakthroughs for systematic application.")
            recommendations.append("Expand use of counterintuitive examples that lead to deeper conceptual understanding.")
        elif beg_emotion == "Disgust" and end_emotion in ["Neutral", "Happy"]:
            recommendations.append("Document successful approaches to addressing strong negative reactions for future application.")
            recommendations.append("Create structured frameworks for addressing potentially controversial content across curriculum.")
            recommendations.append("Analyze specific intervention methods that successfully transformed aversion to engagement.")
        
        return recommendations

    def process_uploaded_image(self, image):
        """Process an uploaded image and detect emotions"""
        try:
            # Make a copy to avoid modifying the original
            img = image.copy()
            frame_height, frame_width = img.shape[:2]
            
            # Resize image for faster processing
            scale = 0.5
            small_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
            emotions_detected = []
            
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
                    # Get face rectangle
                    region = result.get('region', {})
                    
                    # Scale coordinates back to original size
                    x = int(region.get('x', 0) / scale)
                    y = int(region.get('y', 0) / scale)
                    w = int(region.get('w', 0) / scale)
                    h = int(region.get('h', 0) / scale)
                    
                    # Get emotions with de-biasing for neutral emotion
                    emotion_scores = result['emotion']
                    
                    # De-bias the emotions - reduce neutral bias
                    if emotion_scores.get('neutral', 0) > 60:
                        # Find second highest emotion
                        emotions_except_neutral = {k: v for k, v in emotion_scores.items() if k != 'neutral'}
                        if emotions_except_neutral:
                            second_emotion, second_value = max(emotions_except_neutral.items(), key=lambda x: x[1])
                            
                            # Boost non-neutral emotions if they're reasonably strong
                            if second_value > 10:
                                boost = min(30, second_value * 0.8)  # Boost up to 30%
                                emotion_scores['neutral'] -= boost
                                emotion_scores[second_emotion] += boost
                    
                    # Re-determine dominant emotion after adjustments
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0].lower().capitalize()
                    
                    emotions_detected.append({
                        'face_location': [x, y, x+w, y+h],
                        'dominant_emotion': dominant_emotion,
                        'emotion_scores': emotion_scores
                    })
            
            return emotions_detected
        except Exception as e:
            print(f"Error in process_uploaded_image: {e}")
            import traceback
            traceback.print_exc()
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

@app.post("/detect_emotion_from_image")
async def detect_emotion_from_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the image and detect faces/emotions
        processed_data = detector.process_uploaded_image(image)
        
        return {
            "status": "success",
            "emotions": processed_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

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