import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const EmotionDetector = ({ setAnalysisResults, isRecording, setIsRecording }) => {
  const webcamRef = useRef(null);
  const [emotions, setEmotions] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [buttonLock, setButtonLock] = useState(false);
  const captureIntervalRef = useRef(null);

  // Start camera and capture frames
  const startCapturing = () => {
    console.log("Starting camera...");
    setIsCapturing(true);
  };

  // Stop camera and clean up
  const stopCapturing = () => {
    console.log("Stopping camera...");
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setIsCapturing(false);
    setEmotions([]);
  };

  // Handle camera button click
  const handleCameraToggle = () => {
    if (buttonLock) return; // Prevent button spam
    
    setButtonLock(true);
    
    if (isCapturing) {
      stopCapturing();
    } else {
      startCapturing();
    }
    
    // Release button lock after a delay
    setTimeout(() => setButtonLock(false), 1000);
  };

  // Handle recording button click
  const handleRecordingToggle = async () => {
    if (buttonLock) return; // Prevent button spam
    
    setButtonLock(true);
    
    try {
      if (!isRecording) {
        console.log("Starting recording...");
        await axios.post('http://localhost:8000/start_recording', {}, { timeout: 5000 });
        setIsRecording(true);
      } else {
        console.log("Stopping recording...");
        const response = await axios.post('http://localhost:8000/stop_recording', {}, { timeout: 10000 });
        setAnalysisResults(response.data);
        setIsRecording(false);
      }
    } catch (error) {
      console.error("Recording toggle error:", error);
      if (isRecording) {
        setIsRecording(false);
      }
    } finally {
      setTimeout(() => setButtonLock(false), 1000);
    }
  };

  // Capture frame function - run when camera is active
  useEffect(() => {
    if (!isCapturing) return;
    
    console.log("Setting up frame capture...");
    
    // Set up frame capture
    const captureFrame = async () => {
      if (!webcamRef.current) return;
      
      try {
        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;
        
        const base64Image = imageSrc.split(',')[1];
        
        const response = await axios.post(
          'http://localhost:8000/detect_emotion', 
          { image: base64Image },
          { timeout: 3000 }
        );
        
        if (response.data && response.data.emotions) {
          setEmotions(response.data.emotions);
        }
      } catch (error) {
        console.error("Frame capture error:", error);
        // Don't update state on error - just log it
      }
    };
    
    // Initial capture
    captureFrame();
    
    // Set up interval
    captureIntervalRef.current = setInterval(captureFrame, 1000);
    
    // Clean up on effect cleanup
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    };
  }, [isCapturing]); // Only re-run when isCapturing changes

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="emotion-detector">
      <div className="webcam-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={640}
          height={480}
          className="webcam"
        />
        
        {emotions.length > 0 && (
          <div className="emotion-overlay">
            {emotions.map((emotion, index) => (
              <div 
                key={index} 
                className="emotion-box"
                style={{
                  left: emotion.face_location[0],
                  top: emotion.face_location[1],
                  width: emotion.face_location[2] - emotion.face_location[0],
                  height: emotion.face_location[3] - emotion.face_location[1],
                }}
              >
                <div className="emotion-label">
                  {emotion.dominant_emotion}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      <div className="controls">
        <button 
          onClick={handleCameraToggle}
          className={`btn ${isCapturing ? 'btn-danger' : 'btn-primary'}`}
          disabled={buttonLock}
        >
          {isCapturing ? 'Stop Camera' : 'Start Camera'}
        </button>
        
        <button 
          onClick={handleRecordingToggle}
          className={`btn ${isRecording ? 'btn-warning' : 'btn-success'}`}
          disabled={!isCapturing || buttonLock}
        >
          {isRecording ? 'Stop Recording & Analyze' : 'Start Recording Session'}
        </button>
      </div>
      
      {buttonLock && (
        <div className="processing-indicator">
          <div className="spinner"></div>
          <p>Processing request...</p>
        </div>
      )}
    </div>
  );
};

export default EmotionDetector;