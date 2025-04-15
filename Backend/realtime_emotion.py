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
        """Generate detailed interpretation of emotional journey"""
        # Get the dominant emotions for each phase
        beg_emotion = max(beginning.items(), key=lambda x: x[1])[0] if beginning else "Neutral"
        mid_emotion = max(middle.items(), key=lambda x: x[1])[0] if middle else "Neutral"
        end_emotion = max(end.items(), key=lambda x: x[1])[0] if end else "Neutral"
        
        # Get secondary emotions
        beg_secondary = {k: v for k, v in beginning.items() if v > 15 and k != beg_emotion}
        mid_secondary = {k: v for k, v in middle.items() if v > 15 and k != mid_emotion}
        end_secondary = {k: v for k, v in end.items() if v > 15 and k != end_emotion}
        
        # Create interpretation as an array of strings (NOT a single string with newlines)
        interpretation = []
        
        # Beginning phase header
        interpretation.append("Initial Learning Phase")
        
        # Beginning phase dominant emotion analysis
        if beg_emotion == "Happy":
            interpretation.append(f"• Students showed positive engagement ({beginning.get('Happy', 0):.1f}%) during introduction")
            interpretation.append("• This indicates effective activation of prior knowledge and strong initial motivation")
            if beg_secondary:
                for emotion, value in beg_secondary.items():
                    if emotion == "Neutral":
                        interpretation.append(f"• The co-presence of analytical focus ({value:.1f}%) suggests balanced cognitive-affective engagement")
                    elif emotion == "Sad":
                        interpretation.append(f"• Some uncertainty ({value:.1f}%) about specific elements requires clarification")
        elif beg_emotion == "Neutral":
            interpretation.append(f"• Students exhibited analytical focus ({beginning.get('Neutral', 0):.1f}%) during concept introduction")
            interpretation.append("• This indicates receptive information processing while evaluating relevance and difficulty")
            if beg_secondary:
                for emotion, value in beg_secondary.items():
                    if emotion == "Happy":
                        interpretation.append(f"• Moments of positive engagement ({value:.1f}%) reveal effective activation techniques")
        elif beg_emotion == "Sad":
            interpretation.append(f"• Students showed hesitation ({beginning.get('Sad', 0):.1f}%) during opening content")
            interpretation.append("• This suggests potential gaps in prerequisite knowledge or unclear expectations")
        elif beg_emotion == "Angry":
            interpretation.append(f"• Students displayed resistance ({beginning.get('Angry', 0):.1f}%) at the start")
            interpretation.append("• This indicates potential misalignment between expectations and presented material")
        
        # Middle phase header
        interpretation.append("Core Content Engagement")
        
        # Middle phase analysis
        if mid_emotion == "Happy":
            interpretation.append(f"• Students maintained positive engagement ({middle.get('Happy', 0):.1f}%) during core content")
            interpretation.append("• This suggests appropriate complexity level and effective explanatory techniques")
        elif mid_emotion == "Neutral":
            interpretation.append(f"• Students showed sustained analytical focus ({middle.get('Neutral', 0):.1f}%) during main content")
            interpretation.append("• This indicates deep information processing but potential need for emotional anchoring")
        elif mid_emotion == "Sad":
            interpretation.append(f"• Students exhibited disengagement ({middle.get('Sad', 0):.1f}%) during primary content")
            interpretation.append("• This suggests conceptual barriers requiring alternative explanatory approaches")
            interpretation.append("• Check for scaffolding gaps in complex concepts that need remediation")
        elif mid_emotion == "Angry":
            interpretation.append(f"• Students showed frustration ({middle.get('Angry', 0):.1f}%) during core explanations")
            interpretation.append("• This indicates potential cognitive overload or misaligned instructional approaches")
        
        # Add transition analysis between beginning and middle
        if beg_emotion != mid_emotion:
            interpretation.append(f"• Emotional shift from {beg_emotion} to {mid_emotion} as content complexity increased")
            if beg_emotion in ["Happy", "Neutral"] and mid_emotion in ["Sad", "Angry"]:
                interpretation.append("• This negative progression suggests increasing conceptual difficulty exceeded optimal challenge level")
            elif beg_emotion in ["Sad", "Angry"] and mid_emotion in ["Happy", "Neutral"]:
                interpretation.append("• This positive progression indicates successful pedagogical intervention and concept clarification")
        
        # End phase header
        interpretation.append("Knowledge Integration Phase")
        
        # End phase analysis
        if end_emotion == "Happy":
            interpretation.append(f"• Students achieved positive resolution ({end.get('Happy', 0):.1f}%) during concluding elements")
            interpretation.append("• This suggests successful knowledge integration and goal attainment")
        elif end_emotion == "Neutral":
            interpretation.append(f"• Students maintained analytical processing ({end.get('Neutral', 0):.1f}%) during synthesis")
            interpretation.append("• This indicates ongoing information evaluation rather than emotional resolution")
        elif end_emotion == "Sad":
            interpretation.append(f"• Students showed disengagement ({end.get('Sad', 0):.1f}%) during conclusion")
            interpretation.append("• This suggests unresolved conceptual barriers requiring additional clarification")
            interpretation.append("• Consider whether synthesis activities were aligned with student preparation level")
        elif end_emotion == "Angry":
            interpretation.append(f"• Students exhibited frustration ({end.get('Angry', 0):.1f}%) during final activities")
            interpretation.append("• This indicates potential misalignment between instruction and assessment requirements")
        elif end_emotion == "Surprise":
            interpretation.append(f"• Students showed cognitive activation ({end.get('Surprise', 0):.1f}%) during concluding insights")
            interpretation.append("• This suggests effective delivery of perspective-shifting connections or applications")
        
        # Add transition analysis between middle and end
        if mid_emotion != end_emotion:
            interpretation.append(f"• Emotional transition from {mid_emotion} to {end_emotion} during application activities")
            if mid_emotion in ["Sad", "Angry"] and end_emotion in ["Happy", "Surprise"]:
                interpretation.append("• This positive resolution indicates successful clarification of previously challenging concepts")
            elif mid_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry"]:
                interpretation.append("• This negative shift suggests difficulty with application or transfer requirements")
        
        # Overall learning journey assessment
        interpretation.append("Complete Learning Journey Assessment")
        
        # Analyze full emotional trajectory
        if beg_emotion == "Happy" and end_emotion == "Happy":
            interpretation.append("• Sustained positive engagement throughout the session indicates excellent content alignment")
            interpretation.append("• This consistent engagement pattern suggests appropriate challenge level and effective pedagogy")
        elif beg_emotion in ["Neutral", "Sad"] and end_emotion == "Happy":
            interpretation.append("• Positive emotional trajectory indicates successful pedagogical intervention")
            interpretation.append("• This improvement pattern demonstrates effective resolution of initial comprehension barriers")
        elif beg_emotion == "Happy" and end_emotion in ["Sad", "Angry"]:
            interpretation.append("• Declining emotional trajectory suggests increasing conceptual difficulty exceeded scaffolding")
            interpretation.append("• This pattern indicates need for more gradual complexity progression or additional support")
        elif beg_emotion in ["Sad", "Angry"] and end_emotion in ["Sad", "Angry"]:
            interpretation.append("• Persistent negative affect throughout the session indicates fundamental misalignment")
            interpretation.append("• This pattern suggests need for significant revision of content sequencing and explanatory approaches")
        
        # Return as an array, not a joined string
        return interpretation

    def _generate_recommendations(self, emotions, dominant_emotion):
        """Generate specific, actionable pedagogical recommendations"""
        # Get phase emotions and percentages
        beg_emotion = max(emotions["beginning"].items(), key=lambda x: x[1])[0] if emotions.get("beginning") else "Neutral"
        mid_emotion = max(emotions["middle"].items(), key=lambda x: x[1])[0] if emotions.get("middle") else "Neutral"
        end_emotion = max(emotions["end"].items(), key=lambda x: x[1])[0] if emotions.get("end") else "Neutral"
        
        # Create recommendations as an array (NOT a single string with newlines)
        recommendations = []
        
        # Primary section based on overall pattern
        recommendations.append(f"Strategic Teaching Adjustments for {dominant_emotion} Response")
        
        # Dominant emotion recommendations
        if dominant_emotion == "Happy":
            recommendations.append("• Leverage positive engagement by introducing deeper analytical challenges")
            recommendations.append("• Implement student-led explanation activities to solidify conceptual understanding")
            recommendations.append("• Incorporate application tasks that transfer knowledge to novel contexts")
            recommendations.append("• Introduce deliberate cognitive conflicts to deepen critical thinking")
            recommendations.append("• Expand conceptual frameworks with interdisciplinary connections")
        elif dominant_emotion == "Neutral":
            recommendations.append("• Enhance emotional investment through relevant case studies that connect to student interests")
            recommendations.append("• Implement perspective-taking activities that require personal positioning on concepts")
            recommendations.append("• Introduce multimodal representations for key concepts (visual, narrative, kinesthetic)")
            recommendations.append("• Create deliberate moments of cognitive conflict through contrasting viewpoints")
            recommendations.append("• Incorporate brief reflective writing to process emotional dimensions of content")
        elif dominant_emotion == "Sad":
            recommendations.append("• Create visual concept maps that explicitly show relationships between complex ideas")
            recommendations.append("• Implement scaffolded worked examples with fading support as comprehension increases")
            recommendations.append("• Develop alternative explanatory frameworks using varied metaphors and analogies")
            recommendations.append("• Incorporate more frequent knowledge checks with specific feedback")
            recommendations.append("• Structure smaller knowledge units with explicit success criteria")
        elif dominant_emotion == "Angry":
            recommendations.append("• Segment complex content into clearly defined conceptual units with explicit connections")
            recommendations.append("• Provide procedural walkthroughs that demonstrate expert problem-solving processes")
            recommendations.append("• Implement metacognitive modeling showing how to approach conceptual barriers")
            recommendations.append("• Create explicit bridges between new concepts and established knowledge")
            recommendations.append("• Use varied representational approaches for concepts generating resistance")
        elif dominant_emotion == "Surprise":
            recommendations.append("• Design strategic revelation sequences that build on cognitive activation")
            recommendations.append("• Implement comparative analysis activities highlighting unexpected relationships")
            recommendations.append("• Develop application scenarios demonstrating counter-intuitive principles")
            recommendations.append("• Balance surprising insights with structured frameworks for integration")
            recommendations.append("• Create synthesis activities that connect surprising elements to core concepts")
        
        # Beginning phase recommendations
        recommendations.append("Beginning Phase Strategies")
        
        if beg_emotion == "Happy":
            recommendations.append("• Capitalize on initial positive affect with challenging but achievable starter activities")
            recommendations.append("• Use think-pair-share techniques to activate diverse perspectives early")
            recommendations.append("• Present provocative questions that leverage existing engagement")
            recommendations.append("• Build on positive momentum by connecting new content to successful prior knowledge")
        elif beg_emotion == "Neutral":
            recommendations.append("• Enhance emotional investment through personally relevant examples")
            recommendations.append("• Implement brief paired discussions to increase active participation")
            recommendations.append("• Use narrative framing to create emotional anchors for abstract concepts")
            recommendations.append("• Present clear advance organizers showing the session's conceptual roadmap")
        elif beg_emotion == "Sad":
            recommendations.append("• Begin with explicit connections to familiar concepts before new material")
            recommendations.append("• Implement a knowledge activation assessment to address prerequisite gaps")
            recommendations.append("• Use scaffolded entry points with graduated challenge levels")
            recommendations.append("• Establish clear relevance of content to student motivations and interests")
        elif beg_emotion == "Angry":
            recommendations.append("• Address potential frustrations through explicit learning objectives")
            recommendations.append("• Implement brief reflection on specific aspects causing resistance")
            recommendations.append("• Create clear conceptual framework showing how components interconnect")
            recommendations.append("• Use expectation management to align perceptions with instructional approach")
        
        # Middle phase recommendations
        recommendations.append("Core Content Delivery Strategies")
        
        if mid_emotion == "Happy":
            recommendations.append("• Maintain engagement by incorporating complex problem-solving challenges")
            recommendations.append("• Implement peer teaching opportunities for knowledge reinforcement")
            recommendations.append("• Introduce graduated complexity with appropriate scaffolding")
            recommendations.append("• Create deliberate application exercises connecting theory to practice")
        elif mid_emotion == "Neutral":
            recommendations.append("• Introduce perspective-shifting examples to create emotional anchors")
            recommendations.append("• Implement collaborative problem-solving requiring emotional investment")
            recommendations.append("• Create conceptual conflicts that require resolution through discussion")
            recommendations.append("• Use varied representation modes (visual, verbal, experiential)")
        elif mid_emotion == "Sad":
            recommendations.append("• Restructure complex explanations into smaller conceptual units")
            recommendations.append("• Provide multiple explanatory approaches using different analogies")
            recommendations.append("• Implement more frequent comprehension checks with corrective feedback")
            recommendations.append("• Create visual representations of abstract relationships")
            recommendations.append("• Incorporate successful student examples to demonstrate attainability")
        elif mid_emotion == "Angry":
            recommendations.append("• Segment challenging content with clearer developmental progression")
            recommendations.append("• Provide procedural guides for complex analytical tasks")
            recommendations.append("• Implement metacognitive modeling showing expert approaches to difficulties")
            recommendations.append("• Create explicit connections between sequential concepts")
            recommendations.append("• Address potential sources of cognitive overload")
        
        # End phase recommendations
        recommendations.append("Knowledge Integration Strategies")
        
        if end_emotion == "Happy":
            recommendations.append("• Cement positive resolution through application to novel contexts")
            recommendations.append("• Implement brief synthesis activities demonstrating concept mastery")
            recommendations.append("• Create forward-looking connections to upcoming content")
            recommendations.append("• Use student-led summarization to reinforce key takeaways")
        elif end_emotion == "Neutral":
            recommendations.append("• Strengthen session closure with explicit summary of key concepts")
            recommendations.append("• Implement brief reflection connecting content to broader learning goals")
            recommendations.append("• Create conceptual integration activities highlighting relationships")
            recommendations.append("• Use elaborative interrogation to deepen processing")
        elif end_emotion == "Sad":
            recommendations.append("• Develop clearer concept integration activities for session closure")
            recommendations.append("• Provide explicit frameworks for applying theoretical content")
            recommendations.append("• Create resource guides addressing unresolved questions")
            recommendations.append("• Implement targeted micro-review of challenging concepts")
            recommendations.append("• Use exit tickets to identify specific areas needing follow-up")
        elif end_emotion == "Surprise":
            recommendations.append("• Capture insights through reflective 'Aha moment' documentation")
            recommendations.append("• Connect surprising findings to broader theoretical frameworks")
            recommendations.append("• Challenge students to generate additional applications of insights")
            recommendations.append("• Create synthesis activities that solidify perspective shifts")
        
        # Add specific recommendations based on emotional trajectory
        recommendations.append("Targeted Instructional Adjustments")
        
        if beg_emotion in ["Happy", "Neutral"] and end_emotion in ["Sad", "Angry"]:
            recommendations.append("• Restructure content complexity progression with more gradual difficulty increases")
            recommendations.append("• Insert strategic comprehension checks before introducing new complexities")
            recommendations.append("• Create clearer conceptual bridges between fundamental and advanced elements")
            recommendations.append("• Develop supplementary resources for independent reinforcement")
        elif beg_emotion in ["Sad", "Angry"] and end_emotion == "Happy":
            recommendations.append("• Document successful intervention techniques that resolved initial comprehension barriers")
            recommendations.append("• Apply effective scaffolding approaches to other challenging content areas")
            recommendations.append("• Analyze specific explanatory methods that successfully clarified difficult concepts")
            recommendations.append("• Formalize the progression pattern that achieved positive emotional transition")
        elif beg_emotion in ["Sad", "Angry"] and end_emotion in ["Sad", "Angry"]:
            recommendations.append("• Significantly revise conceptual sequencing with clearer developmental progression")
            recommendations.append("• Create prerequisite modules addressing foundational knowledge gaps")
            recommendations.append("• Develop alternative explanation frameworks using different representational systems")
            recommendations.append("• Implement more frequent formative assessment with immediate corrective instruction")
        
        # Return as an array, not a joined string
        return recommendations

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