import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
from keras.utils import img_to_array
import os
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        # Load YOLO model for face detection
        self.yolo_model = YOLO(os.path.join(os.path.dirname(__file__), 'Backend', 'yolov8n.pt'))
        # Load emotion classifier exactly as in the original implementation
        self.emotion_model = load_model(os.path.join(os.path.dirname(__file__), 'Backend', 'Final_model.h5'))
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_emotion(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        results = self.yolo_model(image, classes=0)
        
        detected_faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                face = image[y1:y2, x1:x2]
                
                if face.size > 0:
                    try:
                        # Save face temporarily for DeepFace
                        temp_path = "temp_face.jpg"
                        cv2.imwrite(temp_path, face)
                        
                        # Analyze with DeepFace
                        analysis = DeepFace.analyze(img_path=temp_path, 
                                                   actions=['emotion'],
                                                   enforce_detection=False)
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        # Get dominant emotion
                        emotion = analysis[0]['dominant_emotion']
                        emotion = emotion.title()
                        
                        # Draw on image and add to results
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, emotion, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        detected_faces.append({
                            'box': (x1, y1, x2, y2),
                            'emotion': emotion
                        })
                        
                    except Exception as e:
                        print(f"DeepFace error: {e}")
                        continue
        
        return image, detected_faces

def preprocess_face(face):
    # Ensure consistent preprocessing across all files
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    roi = gray_face.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

def main():
    detector = EmotionDetector()
    
    # Define a list of test images
    image_paths = [
        'img/sad.jpg',
        'img/sad2.jpg',
        'img/sad3.jpg',
        'img/sad4.jpg'
        # Add more images as needed
    ]
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"\nProcessing image: {image_path}")
            result_image, detections = detector.detect_emotion(image_path)
            
            # Display the result image
            cv2.imshow(f'Emotion Detection Result - {i+1}', result_image)
            
            # Save the result image with a unique filename
            output_path = f'result_{i+1}.jpg'
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to {output_path}")
            
            # Print detections in the console
            print(f"Found {len(detections)} faces:")
            for j, face in enumerate(detections, 1):
                print(f"Face {j}: {face['emotion']}")
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Wait for a key press after showing all images
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
