def evaluate_model():
    detector = EmotionDetector()
    test_dir = "FER dataset/test"
    
    results = {emotion: {"correct": 0, "total": 0} for emotion in detector.emotion_labels}
    
    for emotion in detector.emotion_labels:
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            try:
                _, detections = detector.detect_emotion(img_path)
                if detections and detections[0]['confidence'] > 50:
                    predicted = detections[0]['emotion']
                    if predicted == emotion:
                        results[emotion]["correct"] += 1
                    results[emotion]["total"] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Print results
    print("\nEvaluation Results:")
    for emotion, counts in results.items():
        accuracy = (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"{emotion}: {accuracy:.1f}% ({counts['correct']}/{counts['total']})")