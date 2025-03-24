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
        """Generate analysis from recorded emotions"""
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        if not self.emotion_data or len(self.emotion_data) < 3:
            print("No emotions were recorded during the session.")
            return {
                "error": "No emotions were recorded. Please try again and make sure your face is visible.",
                "duration": duration,
                "stats": {"neutral": 100.0}, 
                "emotion_journey": {
                    "beginning": {"neutral": 100.0},
                    "middle": {"neutral": 100.0},
                    "end": {"neutral": 100.0}
                }
            }
        
        # Calculate emotion statistics
        emotion_counts = {}
        for record in self.emotion_data:
            emotion = record["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        total_records = len(self.emotion_data)
        emotion_stats = {emotion: (count / total_records) * 100 for emotion, count in emotion_counts.items()}
        
        # Divide the recording into beginning, middle, and end sections
        section_size = max(1, total_records // 3)
        beginning = self.emotion_data[:section_size]
        middle = self.emotion_data[section_size:section_size*2]
        end = self.emotion_data[section_size*2:]
        
        # Calculate emotions for each section
        beginning_emotions = {}
        middle_emotions = {}
        end_emotions = {}
        
        for record in beginning:
            emotion = record["emotion"]
            if emotion not in beginning_emotions:
                beginning_emotions[emotion] = 0
            beginning_emotions[emotion] += 1
        
        for record in middle:
            emotion = record["emotion"]
            if emotion not in middle_emotions:
                middle_emotions[emotion] = 0
            middle_emotions[emotion] += 1
        
        for record in end:
            emotion = record["emotion"]
            if emotion not in end_emotions:
                end_emotions[emotion] = 0
            end_emotions[emotion] += 1
        
        # Convert to percentages
        beginning_stats = {e: (c / len(beginning)) * 100 for e, c in beginning_emotions.items()} if beginning else {"neutral": 100}
        middle_stats = {e: (c / len(middle)) * 100 for e, c in middle_emotions.items()} if middle else {"neutral": 100}
        end_stats = {e: (c / len(end)) * 100 for e, c in end_emotions.items()} if end else {"neutral": 100}
        
        # Generate analysis
        dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1])[0] if emotion_stats else "neutral"
        
        return {
            "duration": duration,
            "stats": emotion_stats,
            "dominant_emotion": dominant_emotion,
            "emotion_journey": {
                "beginning": beginning_stats,
                "middle": middle_stats,
                "end": end_stats
            },
            "interpretation": f"The dominant emotion was {dominant_emotion}.",
            "educational_recommendations": [
                "Consider how your emotions might affect learning outcomes.",
                "Being aware of your emotional state can help improve focus."
            ]
        }