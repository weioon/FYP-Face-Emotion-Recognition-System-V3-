import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self):
        # Load YOLO model for face detection
        self.yolo_model = YOLO('yolov8n.pt')
        # Load emotion classifier exactly as in the original implementation
        self.emotion_model = load_model('Final_model.h5')
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
                    # Preprocess the face exactly as in the original implementation
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (48, 48))
                    roi = gray_face.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Predict emotion
                    prediction = self.emotion_model.predict(roi)[0]
                    emotion_label = self.emotion_labels[prediction.argmax()]
                    
                    # Draw rectangle and emotion label on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, emotion_label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    detected_faces.append({
                        'box': (x1, y1, x2, y2),
                        'emotion': emotion_label
                    })
        
        return image, detected_faces

def main():
    detector = EmotionDetector()
    image_path = 'surprise2.jpg'  # Replace with your test image path
    
    try:
        result_image, detections = detector.detect_emotion(image_path)
        
        # Display the result image
        cv2.imshow('Emotion Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the result image
        cv2.imwrite('result.jpg', result_image)
        
        # Print detections in the console
        print(f"Found {len(detections)} faces:")
        for i, face in enumerate(detections, 1):
            print(f"Face {i}: {face['emotion']}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
