import cv2
import numpy as np
import time
from deepface import DeepFace

class LightweightDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recording = False
        self.start_time = None
        self.end_time = None
        self.emotion_data = []
        self.current_emotion_data = []
        self.last_full_analysis_time = 0
        self.cached_emotions = []
        self.analysis_interval = 2.0  # seconds

    def detect_faces_only(self, frame):
        """Just detect face locations without emotion analysis (very fast)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            # Use cached emotion if available, otherwise use neutral
            face_data = {
                "face_location": [x, y, x+w, y+h],
                "dominant_emotion": "neutral",
                "emotion_scores": {"neutral": 100.0}
            }
            results.append(face_data)
            
        self.current_emotion_data = results
        return results

    def detect_emotions_lightweight(self, frame):
        """Detect emotions, but with optimizations for speed"""
        current_time = time.time()
        
        # First just detect faces (fast operation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If no faces or not time for full analysis yet, return face locations only
        if len(faces) == 0 or (current_time - self.last_full_analysis_time < self.analysis_interval):
            face_data = []
            for i, (x, y, w, h) in enumerate(faces):
                # Use cached emotion if available, otherwise use neutral
                if i < len(self.cached_emotions):
                    cached = self.cached_emotions[i]
                    cached["face_location"] = [x, y, x+w, y+h]  # Update position
                    face_data.append(cached)
                else:
                    face_data.append({
                        "face_location": [x, y, x+w, y+h],
                        "dominant_emotion": "neutral",
                        "emotion_scores": {"neutral": 100.0}
                    })
            
            self.current_emotion_data = face_data
            return face_data
        
        # It's time for a full analysis
        self.last_full_analysis_time = current_time
        
        # Process smaller image for speed (30% of original size)
        scale = 0.3
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        face_data = []
        try:
            results = DeepFace.analyze(
                small_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results]
                
            for result in results:
                if 'emotion' in result and 'region' in result:
                    # Adjust coordinates back to original size
                    x = int(result['region']['x'] / scale)
                    y = int(result['region']['y'] / scale)
                    w = int(result['region']['w'] / scale)
                    h = int(result['region']['h'] / scale)
                    
                    emotions_dict = result['emotion']
                    dominant_emotion = result['dominant_emotion']
                    
                    face_data.append({
                        "face_location": [x, y, x+w, y+h],
                        "dominant_emotion": dominant_emotion,
                        "emotion_scores": emotions_dict
                    })
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            # On error, return just face locations
            for (x, y, w, h) in faces:
                face_data.append({
                    "face_location": [x, y, x+w, y+h],
                    "dominant_emotion": "neutral",
                    "emotion_scores": {"neutral": 100.0}
                })
        
        self.cached_emotions = face_data
        self.current_emotion_data = face_data
        
        # Add to recording data if recording
        if self.recording and face_data:
            timestamp = time.time() - self.start_time if self.start_time else 0
            for face in face_data:
                self.emotion_data.append({
                    "timestamp": timestamp,
                    "emotion": face["dominant_emotion"],
                    "score": face["emotion_scores"].get(face["dominant_emotion"].lower(), 0)
                })
                
        return face_data
    
    def start_recording(self):
        """Start recording emotions"""
        self.recording = True
        self.start_time = time.time()
        self.emotion_data = []
        print("Started recording")

    def stop_recording(self):
        """Stop recording emotions"""
        if not self.recording:
            return
        
        self.recording = False
        self.end_time = time.time()
        print(f"Stopped recording. Collected {len(self.emotion_data)} data points")

    def analyze_emotions(self):
        """Analyze recorded emotion data and provide insights"""
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # If no emotions were recorded, return a helpful message with fallback data
        if not self.emotion_data or len(self.emotion_data) < 3:
            print("No emotions were recorded during the session.")
            default_data = {
                "warning": "Limited emotion data recorded",
                "duration": duration,
                "stats": {"neutral": 100.0},
                "dominant_emotion": "neutral",
                "significant_emotions": [
                    {"emotion": "neutral", "percentage": 100.0}
                ],
                "emotion_journey": {
                    "beginning": {"neutral": 100.0},
                    "middle": {"neutral": 100.0},
                    "end": {"neutral": 100.0}
                },
                "summary": "No significant emotions detected during this session.",
                "interpretation": "Insufficient data was captured to provide a meaningful interpretation of emotional patterns.",
                "educational_recommendations": [
                    "Ensure your face is clearly visible to the camera",
                    "Check lighting conditions for better facial detection",
                    "Try to maintain your position within the camera frame"
                ],
                "recommendations": [
                    "Try to position yourself better in front of the camera",
                    "Ensure good lighting for better emotion detection"
                ]
            }
            return default_data
        
        # Process the emotion data
        emotion_counts = {}
        for record in self.emotion_data:
            emotion = record["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        total_records = len(self.emotion_data)
        emotion_stats = {emotion: (count / total_records) * 100 for emotion, count in emotion_counts.items()}
        
        # Get dominant emotion
        dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1])[0] if emotion_stats else "neutral"
        
        # Create significant emotions list for frontend display
        significant_emotions = []
        for emotion, percentage in emotion_stats.items():
            if percentage >= 5.0:  # Only include emotions that are at least 5% of the time
                significant_emotions.append({
                    "emotion": emotion,
                    "percentage": percentage
                })
        
        # Sort by percentage
        significant_emotions.sort(key=lambda x: x["percentage"], reverse=True)
        
        # Divide the recording into beginning, middle, and end sections
        section_size = max(1, total_records // 3)
        beginning = self.emotion_data[:section_size]
        middle = self.emotion_data[section_size:section_size*2]
        end = self.emotion_data[section_size*2:]
        
        # Calculate emotions for each section
        beginning_emotions = self._calculate_section_emotions(beginning)
        middle_emotions = self._calculate_section_emotions(middle)
        end_emotions = self._calculate_section_emotions(end)
        
        # Generate analysis
        summary = f"During this session, the dominant emotion was {dominant_emotion} ({emotion_stats.get(dominant_emotion, 0):.1f}%)."
        
        # Generate recommendations based on emotions
        recommendations = [
            f"Your dominant emotion was {dominant_emotion}. Consider how this might affect your learning.",
            "Being aware of your emotional state can help improve focus and learning outcomes."
        ]
        
        if "happy" in emotion_stats and emotion_stats["happy"] > 30:
            recommendations.append("You seemed happy during this session, which is great for learning!")
        elif "sad" in emotion_stats and emotion_stats["sad"] > 30:
            recommendations.append("You seemed sad during this session. Consider taking breaks to improve mood.")
        elif "angry" in emotion_stats and emotion_stats["angry"] > 20:
            recommendations.append("Detected frustration during this session. Consider breaking complex tasks into smaller steps.")
        
        # Generate interpretation
        interpretation = self._interpret_emotional_journey(
            {"beginning": beginning_emotions, "middle": middle_emotions, "end": end_emotions},
            emotion_stats
        )

        # Generate educational recommendations
        if "happy" in emotion_stats and emotion_stats["happy"] > 30:
            educational_recommendations = [
                "The positive engagement suggests this teaching approach was effective.",
                "Consider building on these concepts in future sessions.",
                "The student's positive response indicates good understanding."
            ]
        elif "sad" in emotion_stats and emotion_stats["sad"] > 30:
            educational_recommendations = [
                "Consider reviewing the material with alternative explanations.",
                "Break complex topics into smaller, more digestible segments.",
                "Additional examples or visual aids might improve understanding."
            ]
        elif "angry" in emotion_stats and emotion_stats["angry"] > 20:
            educational_recommendations = [
                "Identify challenging sections that may have caused frustration.",
                "Consider alternative approaches to the difficult concepts.",
                "Provide additional practice exercises for these topics."
            ]
        else:
            educational_recommendations = [
                "Maintain a balance of challenge and achievement in learning materials.",
                "Consider varying teaching methods to engage different learning styles.",
                "Regular check-ins can help identify areas needing clarification."
            ]

        return {
            "duration": duration,
            "stats": emotion_stats,
            "dominant_emotion": dominant_emotion,
            "significant_emotions": significant_emotions,
            "emotion_journey": {
                "beginning": beginning_emotions,
                "middle": middle_emotions,
                "end": end_emotions
            },
            "summary": summary,
            "interpretation": interpretation,
            "educational_recommendations": educational_recommendations,
            "recommendations": recommendations
        }

    # Helper method for section emotion calculation
    def _calculate_section_emotions(self, section_data):
        """Calculate emotion percentages for a section of the recording"""
        if not section_data:
            return {"neutral": 100.0}
            
        emotion_counts = {}
        for record in section_data:
            emotion = record["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        total_records = len(section_data)
        return {e: (c / total_records) * 100 for e, c in emotion_counts.items()}

    # Add this method to LightweightDetector class
    def _interpret_emotional_journey(self, emotion_journey, emotion_stats):
        """Generate an interpretation of the emotional journey"""
        interpretation = "Based on the emotional patterns observed: "
        
        # Analyze beginning state
        if emotion_journey["beginning"]:
            beginning_emotions = emotion_journey["beginning"]
            interpretation += "\n\nAt the beginning: "
            
            if "happy" in beginning_emotions and beginning_emotions["happy"] > 25:
                interpretation += "The student started with a positive mindset, showing engagement and readiness to learn. "
            if "neutral" in beginning_emotions and beginning_emotions["neutral"] > 40:
                interpretation += "The student began in a calm, receptive state. "
            if "angry" in beginning_emotions and beginning_emotions["angry"] > 15:
                interpretation += "The student may have started with frustration or resistance to the subject matter. "
            if "sad" in beginning_emotions and beginning_emotions["sad"] > 15:
                interpretation += "The student showed signs of disengagement or concern at the start. "
        
        # Analyze middle state
        if emotion_journey["middle"]:
            middle_emotions = emotion_journey["middle"]
            interpretation += "\n\nDuring the middle: "
            
            if "happy" in middle_emotions and middle_emotions["happy"] > 25:
                interpretation += "The student experienced moments of understanding or connection with the material. "
            if "sad" in middle_emotions and middle_emotions["sad"] > 15:
                interpretation += "The student may have struggled with comprehension or engagement. "
            if "angry" in middle_emotions and middle_emotions["angry"] > 15:
                interpretation += "The student encountered concepts that were challenging or frustrating. "
        
        # Analyze end state
        if emotion_journey["end"]:
            end_emotions = emotion_journey["end"]
            interpretation += "\n\nBy the end: "
            
            if "happy" in end_emotions and end_emotions["happy"] > 25:
                interpretation += "The student achieved a sense of accomplishment or satisfaction. "
            if "neutral" in end_emotions and end_emotions["neutral"] > 50:
                interpretation += "The student maintained composed attention through the conclusion. "
            if "sad" in end_emotions and end_emotions["sad"] > 20:
                interpretation += "The student may have felt disappointed or unconfident about their learning outcome. "
        
        # Analyze overall pattern
        dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1])[0] if emotion_stats else "neutral"
        interpretation += f"\n\nOverall, the predominant emotion was {dominant_emotion}, which suggests "
        
        if dominant_emotion == "happy":
            interpretation += "a positive learning experience with good engagement."
        elif dominant_emotion == "neutral":
            interpretation += "focused attention, though this could also indicate passive engagement."
        elif dominant_emotion == "sad":
            interpretation += "some difficulties with the material or potential disengagement."
        elif dominant_emotion == "angry":
            interpretation += "frustration with challenging concepts that may need more explanation."
        else:
            interpretation += "various emotional responses to the learning material."
        
        return interpretation