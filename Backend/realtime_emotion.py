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
            
            # Get secondary emotions (>15%) for more nuanced analysis
            beginning_secondary = [e for e, v in beginning.items() if v > 15 and e != beginning_emotion]
            middle_secondary = [e for e, v in middle.items() if v > 15 and e != middle_emotion]
            end_secondary = [e for e, v in end.items() if v > 15 and e != end_emotion]
            
            interpretation = []
            
            # Beginning phase analysis
            interpretation.append("## Initial Learning Phase")
            
            # Primary emotion analysis
            if beginning_emotion == "Happy":
                interpretation.append("• Students demonstrated positive engagement during initial content introduction")
                interpretation.append("• This receptiveness indicates effective activation of prior knowledge and successful establishment of learning objectives")
            elif beginning_emotion == "Neutral":
                interpretation.append("• Students exhibited focused attention during concept introduction")
                interpretation.append("• This analytical stance suggests information processing with reserved judgment about the new material")
            elif beginning_emotion == "Sad":
                interpretation.append("• Students showed signs of uncertainty during the opening segment")
                interpretation.append("• This hesitancy suggests potential gaps in prerequisite knowledge needed for the new material")
            elif beginning_emotion == "Angry":
                interpretation.append("• Students exhibited signs of cognitive resistance during the introduction")
                interpretation.append("• This frustration indicates potential misalignment between student expectations and presented content")
            elif beginning_emotion == "Surprise":
                interpretation.append("• Students displayed heightened alertness during initial concept introduction")
                interpretation.append("• This response suggests effective use of counter-intuitive examples or perspective-challenging questions")
            
            # Add nuance with secondary emotions
            if beginning_secondary:
                interpretation.append(f"• While primarily {beginning_emotion.lower()}, students also showed notable {'/'.join(beginning_secondary).lower()} responses")
                if "Happy" in beginning_secondary and beginning_emotion != "Happy":
                    interpretation.append("• These moments of positive affect suggest particular content elements resonated well despite overall challenges")
                if "Sad" in beginning_secondary and beginning_emotion != "Sad":
                    interpretation.append("• These moments of hesitation indicate specific conceptual areas requiring additional clarification")
                if "Surprise" in beginning_secondary:
                    interpretation.append("• These moments of cognitive activation indicate effective use of perspective-shifting examples")
            
            # Middle phase analysis with transition description
            interpretation.append("\n## Core Content Engagement")
            
            # Analyze emotional transitions
            if beginning_emotion != middle_emotion:
                interpretation.append(f"• A clear transition from {beginning_emotion.lower()} to {middle_emotion.lower()} occurred during primary content delivery")
                
                # Analyze specific transition patterns
                if beginning_emotion in ["Neutral", "Sad"] and middle_emotion == "Happy":
                    interpretation.append("• This positive shift indicates successful scaffolding that bridged initial knowledge gaps")
                    interpretation.append("• The transition suggests explanatory approaches effectively addressed student uncertainties")
                elif beginning_emotion in ["Happy", "Neutral"] and middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This downward shift points to specific challenges with complex concepts in the middle section")
                    interpretation.append("• The change suggests the need to revisit the sequencing or scaffolding of advanced material")
                elif beginning_emotion in ["Sad", "Angry"] and middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• The persistent challenge indicators suggest unresolved conceptual barriers")
                    interpretation.append("• This pattern typically occurs when foundational misunderstandings remain unaddressed")
            else:
                interpretation.append(f"• The {middle_emotion.lower()} engagement pattern remained consistent during the primary content")
                
                if middle_emotion == "Happy":
                    interpretation.append("• This sustained positive engagement suggests appropriate complexity progression")
                    interpretation.append("• The consistency indicates effective alignment between content difficulty and student capability")
                elif middle_emotion == "Neutral":
                    interpretation.append("• This maintained analytical focus suggests deep information processing")
                    interpretation.append("• Consider that while attentive, students might benefit from emotional investment opportunities")
                elif middle_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This continued difficulty suggests fundamental misalignment in conceptual approach")
                    interpretation.append("• The persistence indicates need for alternative explanatory frameworks or prerequisites")
            
            # Add nuance with secondary emotions in middle section
            if middle_secondary:
                interpretation.append(f"• Beyond the primary {middle_emotion.lower()} response, students showed significant {'/'.join(middle_secondary).lower()} reactions during core content")
                if "Happy" in middle_secondary and middle_emotion not in ["Happy"]:
                    interpretation.append("• These positive moments identify specific explanations or examples that effectively resonated")
                if "Surprise" in middle_secondary:
                    interpretation.append("• These moments of heightened attention mark conceptual breakthroughs or perspective shifts")
                if ("Sad" in middle_secondary or "Angry" in middle_secondary) and middle_emotion not in ["Sad", "Angry"]:
                    interpretation.append("• These challenging moments highlight specific content segments requiring additional support")
            
            # End phase analysis
            interpretation.append("\n## Knowledge Integration Phase")
            
            if middle_emotion != end_emotion:
                interpretation.append(f"• The concluding segment showed a transition from {middle_emotion.lower()} to {end_emotion.lower()}")
                
                if middle_emotion in ["Sad", "Angry"] and end_emotion == "Happy":
                    interpretation.append("• This positive resolution indicates successful clarification of previously challenging concepts")
                    interpretation.append("• The pattern suggests effective summary techniques that integrated fragmented knowledge")
                elif middle_emotion == "Happy" and end_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This decline suggests difficulty with synthesis or application requirements")
                    interpretation.append("• This pattern often emerges when conceptual understanding faces application challenges")
                elif middle_emotion in ["Happy", "Neutral"] and end_emotion in ["Neutral"]:
                    interpretation.append("• This shift to analytical processing suggests appropriate consolidation activities")
            else:
                interpretation.append(f"• The {end_emotion.lower()} response continued through the conclusion")
                
                if end_emotion == "Happy":
                    interpretation.append("• This sustained positive affect suggests successful knowledge integration")
                    interpretation.append("• Students likely achieved the learning objectives with confidence in their understanding")
                elif end_emotion == "Neutral":
                    interpretation.append("• This continued analytical stance suggests ongoing information processing")
                    interpretation.append("• Consider whether more explicit closure would enhance learning consolidation")
                elif end_emotion in ["Sad", "Angry"]:
                    interpretation.append("• This persistent challenge suggests unresolved questions remain at conclusion")
                    interpretation.append("• Students likely need additional reinforcement or alternative approaches")
            
            # Add nuance with secondary emotions in end section
            if end_secondary:
                interpretation.append(f"• While primarily {end_emotion.lower()}, the conclusion also elicited {'/'.join(end_secondary).lower()} responses")
                if "Happy" in end_secondary and end_emotion not in ["Happy"]:
                    interpretation.append("• These positive moments highlight specific synthesis activities that resonated effectively")
                if "Sad" in end_secondary and end_emotion not in ["Sad"]:
                    interpretation.append("• These moments of uncertainty identify specific areas requiring follow-up reinforcement")
            
            # Overall learning trajectory assessment
            interpretation.append("\n## Complete Learning Journey Assessment")
            
            # Analyze full emotional arc
            if beginning_emotion == "Neutral" and middle_emotion == "Happy" and end_emotion == "Happy":
                interpretation.append("• This progression from analytical focus to sustained engagement represents ideal learning activation")
                interpretation.append("• This pattern indicates excellent alignment between teaching approach and student cognitive needs")
            elif beginning_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry"]:
                interpretation.append("• This declining trajectory suggests cumulative cognitive load exceeded processing capacity")
                interpretation.append("• This pattern indicates need for additional scaffolding as content complexity increases")
            elif beginning_emotion in ["Sad", "Angry"] and end_emotion in ["Happy", "Neutral"]:
                interpretation.append("• This positive progression demonstrates effective resolution of conceptual barriers")
                interpretation.append("• Your teaching strategies successfully addressed initial comprehension challenges")
            elif beginning_emotion == end_emotion == "Happy":
                interpretation.append("• This sustained positive engagement indicates excellent content alignment with student capabilities")
                interpretation.append("• The consistent pattern suggests effective teaching strategies throughout the session")
            elif beginning_emotion == end_emotion == "Neutral":
                interpretation.append("• This consistent analytical processing suggests information-dense content requiring active processing")
                interpretation.append("• Consider adding emotional anchors to enhance memory formation and engagement")
            elif beginning_emotion == end_emotion == "Sad" or beginning_emotion == end_emotion == "Angry":
                interpretation.append("• This persistent negative affect indicates fundamental misalignment requiring significant revision")
                interpretation.append("• Consider restructuring content sequence or reinforcing prerequisite knowledge")
            else:
                interpretation.append("• The varied emotional pattern reveals a complex learning journey with distinct challenges")
                interpretation.append("• This variability suggests need for targeted intervention at specific content junctures")
            
            return "\n".join(interpretation)
        except Exception as e:
            print(f"Error in interpretation generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return "• Insufficient data to provide a detailed emotional journey analysis"

    def _generate_recommendations(self, emotions, dominant_emotion):
        """Generate specific, actionable pedagogical recommendations using bullet points"""
        try:
            recommendations = []
            
            # Get emotions from each stage
            beg_emotion = max(emotions["beginning"].items(), key=lambda x: x[1])[0] if emotions.get("beginning") else "Neutral"
            mid_emotion = max(emotions["middle"].items(), key=lambda x: x[1])[0] if emotions.get("middle") else "Neutral"
            end_emotion = max(emotions["end"].items(), key=lambda x: x[1])[0] if emotions.get("end") else "Neutral"
            
            # Check for secondary emotions (>15%) for more nuanced recommendations
            all_emotions = {}
            for phase in ["beginning", "middle", "end"]:
                for emotion, value in emotions.get(phase, {}).items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + value
                    
            # Calculate average percentages across phases
            total_phases = 3
            avg_emotions = {e: v/total_phases for e, v in all_emotions.items()}
            secondary_emotions = [e for e, v in avg_emotions.items() if v > 15 and e != dominant_emotion]
            
            # Primary recommendations based on dominant emotion pattern
            recommendations.append(f"## Strategic Teaching Adjustments for {dominant_emotion} Response")
            
            if dominant_emotion == "Happy":
                recommendations.append("### Leveraging Positive Engagement")
                recommendations.append("• Integrate problem-based learning activities that apply established concepts to novel contexts")
                recommendations.append("• Implement structured peer teaching opportunities where students articulate complex concepts to classmates")
                recommendations.append("• Introduce deliberate cognitive challenges that push understanding to deeper analytical levels")
                
                # Add nuance based on secondary emotions
                if "Neutral" in secondary_emotions:
                    recommendations.append("• Balance conceptual exploration with structured analytical frameworks to support both engaged and analytical learners")
                if "Surprise" in secondary_emotions:
                    recommendations.append("• Maintain the successful use of perspective-shifting examples that created moments of insight")
                if "Sad" in secondary_emotions or "Angry" in secondary_emotions:
                    recommendations.append("• Address specific challenging segments by providing alternative explanatory frameworks")
                
            elif dominant_emotion == "Neutral":
                recommendations.append("### Enhancing Analytical Engagement")
                recommendations.append("• Incorporate case studies that create emotional connections to abstract theoretical concepts")
                recommendations.append("• Implement brief perspective-taking activities that require personal positioning on conceptual debates")
                recommendations.append("• Integrate multimedia elements that provide alternative representational forms for key concepts")
                
                # Add nuance based on secondary emotions
                if "Happy" in secondary_emotions:
                    recommendations.append("• Expand upon the specific elements that generated positive responses by applying similar approaches to other content")
                if "Sad" in secondary_emotions:
                    recommendations.append("• Provide additional conceptual scaffolding focused on the specific content areas that created uncertainty")
                if "Surprise" in secondary_emotions:
                    recommendations.append("• Develop more counterintuitive examples that achieved cognitive activation during the session")
                
            elif dominant_emotion == "Sad":
                recommendations.append("### Addressing Comprehension Barriers")
                recommendations.append("• Implement periodic concept mapping activities to visually organize relationships between key ideas")
                recommendations.append("• Develop worked examples that demonstrate expert problem-solving processes with explicit reasoning")
                recommendations.append("• Create retrieval practice opportunities that reinforce foundational concepts through application")
                
                # Add nuance based on secondary emotions
                if "Happy" in secondary_emotions:
                    recommendations.append("• Identify and expand upon the specific content elements that generated positive responses")
                if "Neutral" in secondary_emotions:
                    recommendations.append("• Maintain the aspects that supported analytical processing while addressing areas of confusion")
                if "Angry" in secondary_emotions:
                    recommendations.append("• Address sources of frustration by providing alternative explanatory approaches for complex concepts")
                
            elif dominant_emotion == "Angry":
                recommendations.append("### Resolving Cognitive Friction")
                recommendations.append("• Restructure complex material into smaller conceptual units with explicit connections between segments")
                recommendations.append("• Implement procedural walkthroughs that demonstrate step-by-step approaches to challenging problems")
                recommendations.append("• Create conceptual bridging activities that connect new material to firmly established knowledge")
                
                # Add nuance based on secondary emotions
                if "Happy" in secondary_emotions:
                    recommendations.append("• Expand upon the specific approaches that resonated positively despite overall challenges")
                if "Surprise" in secondary_emotions:
                    recommendations.append("• Leverage the moments of cognitive breakthrough by expanding similar techniques to challenging areas")
                
            elif dominant_emotion == "Surprise":
                recommendations.append("### Building on Cognitive Activation")
                recommendations.append("• Develop sequential reveal activities that deliberately challenge initial assumptions before resolution")
                recommendations.append("• Implement comparative analysis tasks that highlight unexpected relationships between concepts")
                recommendations.append("• Create application scenarios that demonstrate counter-intuitive outcomes of theoretical principles")
                
                # Add nuance based on secondary emotions
                if "Happy" in secondary_emotions:
                    recommendations.append("• Balance cognitive challenge with positive reinforcement to maintain engagement")
                if "Neutral" in secondary_emotions:
                    recommendations.append("• Follow surprising elements with structured analytical activities to consolidate understanding")
                
            # Add recommendations based on emotional journey pattern
            recommendations.append("\n### Session Structure Optimization")
            
            # Analyze specific emotional progression patterns
            if beg_emotion in ["Sad", "Angry"] and end_emotion in ["Sad", "Angry"]:
                recommendations.append("• Develop pre-session conceptual primers that establish essential prerequisite knowledge")
                recommendations.append("• Restructure the content sequence to provide more gradual complexity progression")
                recommendations.append("• Implement frequent comprehension checkpoints with specific corrective instruction")
                
            elif beg_emotion in ["Neutral", "Sad"] and end_emotion == "Happy":
                recommendations.append("• Replicate the successful scaffolding approach that transformed initial uncertainty to understanding")
                recommendations.append("• Identify specific conceptual bridges that facilitated breakthrough moments")
                recommendations.append("• Begin future sessions with concrete examples before introducing abstract principles")
                
            elif beg_emotion == "Happy" and end_emotion in ["Sad", "Angry"]:
                recommendations.append("• Redistribute cognitive load more evenly throughout the session")
                recommendations.append("• Insert periodic application exercises before introducing additional theoretical complexity")
                recommendations.append("• Create explicit connections between sequential concepts to prevent fragmented understanding")
                
            elif beg_emotion == "Neutral" and mid_emotion == "Happy" and end_emotion == "Happy":
                recommendations.append("• Document the specific progressive disclosure techniques that achieved this effective pattern")
                recommendations.append("• Analyze the specific explanatory approaches that transformed attention into engagement")
                recommendations.append("• Apply similar scaffolding sequences to other complex content areas")
            
            # Phase-specific recommendations
            recommendations.append("\n### Phase-Specific Teaching Strategies")
            
            # Beginning phase
            if "beginning" in emotions and emotions["beginning"]:
                if "Happy" in emotions["beginning"] and emotions["beginning"]["Happy"] > 40:
                    recommendations.append("**Introduction Techniques:**")
                    recommendations.append("• Maintain your effective activation strategies that establish initial engagement")
                    recommendations.append("• Consider beginning with brief application scenarios that connect to student experience")
                    
                elif "Neutral" in emotions["beginning"] and emotions["beginning"]["Neutral"] > 50:
                    recommendations.append("**Introduction Techniques:**")
                    recommendations.append("• Strengthen emotional investment by incorporating relevant case studies or scenarios")
                    recommendations.append("• Implement brief paired discussion activities to increase active participation")
                    
                elif any(emotions["beginning"].get(e, 0) > 30 for e in ["Sad", "Angry"]):
                    recommendations.append("**Introduction Techniques:**")
                    recommendations.append("• Begin with explicit connections to familiar concepts before introducing new material")
                    recommendations.append("• Implement structured activation activities that assess and address prerequisite knowledge")
                    
            # Middle phase
            if "middle" in emotions and emotions["middle"]:
                if "Happy" in emotions["middle"] and emotions["middle"]["Happy"] > 40:
                    recommendations.append("**Core Content Techniques:**")
                    recommendations.append("• Maintain your effective explanatory approaches during complex content delivery")
                    recommendations.append("• Consider implementing student-led explanation opportunities to deepen understanding")
                    
                elif "Neutral" in emotions["middle"] and emotions["middle"]["Neutral"] > 50:
                    recommendations.append("**Core Content Techniques:**")
                    recommendations.append("• Incorporate more collaborative problem-solving activities for complex concepts")
                    recommendations.append("• Integrate real-world applications that create emotional connection to abstract ideas")
                    
                elif any(emotions["middle"].get(e, 0) > 30 for e in ["Sad", "Angry"]):
                    recommendations.append("**Core Content Techniques:**")
                    recommendations.append("• Break complex explanations into smaller conceptual units with explicit connections")
                    recommendations.append("• Provide alternative representational approaches for challenging concepts (visual, narrative, etc.)")
                    
            # End phase
            if "end" in emotions and emotions["end"]:
                if "Happy" in emotions["end"] and emotions["end"]["Happy"] > 40:
                    recommendations.append("**Conclusion Techniques:**")
                    recommendations.append("• Maintain your effective synthesis activities that consolidated understanding")
                    recommendations.append("• Consider implementing brief application exercises that demonstrate concept mastery")
                    
                elif "Neutral" in emotions["end"] and emotions["end"]["Neutral"] > 50:
                    recommendations.append("**Conclusion Techniques:**")
                    recommendations.append("• Implement clearer closure activities that explicitly summarize key takeaways")
                    recommendations.append("• Create brief reflection opportunities that connect content to broader course objectives")
                    
                elif any(emotions["end"].get(e, 0) > 30 for e in ["Sad", "Angry"]):
                    recommendations.append("**Conclusion Techniques:**")
                    recommendations.append("• Develop concept integration activities that clarify relationships between complex ideas")
                    recommendations.append("• Provide clear pathways for follow-up support on challenging material")
            
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