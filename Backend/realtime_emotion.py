import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from ultralytics import YOLO

class RealtimeEmotionDetector:
    def __init__(self):
        # Load YOLO model for face detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Load the emotion recognition model
        self.emotion_model = load_model('Final_model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def process_frame(self, frame):
        results = self.yolo_model(frame, classes=0)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                face = frame[y1:y2, x1:x2]
                
                if face.size > 0:
                    # Match exact preprocessing from original
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (48, 48))
                    roi = gray_face.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Use model for prediction
                    prediction = self.emotion_model.predict(roi)[0]
                    emotion = self.emotion_labels[prediction.argmax()]
                    confidence = prediction.max() * 100
                    
                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

def main():
    detector = RealtimeEmotionDetector()
    cap = cv2.VideoCapture(0)  # Use default webcam (0)
    
    print("Starting real-time emotion detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        result_frame = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Real-time Emotion Detection', result_frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()