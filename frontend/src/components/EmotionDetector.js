import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import Card from './Card';
import Button from './Button';

const EmotionDetector = ({ setAnalysisResults, isRecording, setIsRecording }) => {
  const webcamRef = useRef(null);
  const [emotions, setEmotions] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [buttonLock, setButtonLock] = useState(false);
  const captureIntervalRef = useRef(null);
  const [debugInfo, setDebugInfo] = useState('');

  // Handle camera capture toggle
  const handleCameraToggle = () => {
    setButtonLock(true);
    
    if (isCapturing) {
      // Stop the camera
      clearInterval(captureIntervalRef.current);
      setEmotions([]);
      setIsCapturing(false);
    } else {
      // Start the camera
      setIsCapturing(true);
      captureIntervalRef.current = setInterval(() => {
        captureFrame();
      }, 1000); // Capture frame every second
    }
    
    setTimeout(() => setButtonLock(false), 1000); // Prevent rapid clicking
  };

  // Handle recording toggle
  const handleRecordingToggle = async () => {
    setButtonLock(true);
    
    if (isRecording) {
      // Stop recording
      try {
        const response = await axios.post('http://localhost:8000/stop_recording/');
        setAnalysisResults(response.data);
        setIsRecording(false);
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = setInterval(() => {
          captureFrame();
        }, 1000); // Back to normal capture rate
      } catch (err) {
        console.error("Error stopping recording:", err);
        setDebugInfo(`Error stopping recording: ${err.message}`);
      }
    } else {
      // Start recording
      try {
        await axios.post('http://localhost:8000/start_recording/');
        setIsRecording(true);
        // Start sending frames more frequently for recording
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = setInterval(() => {
          captureFrame(true); // true indicates recording mode
        }, 200); // More frequent captures during recording
      } catch (err) {
        console.error("Error starting recording:", err);
        setDebugInfo(`Error starting recording: ${err.message}`);
      }
    }
    
    setTimeout(() => setButtonLock(false), 1000); // Prevent rapid clicking
  };

  // Capture a frame from the webcam
  const captureFrame = async (isRecordingMode = false) => {
    if (!webcamRef.current) {
      setDebugInfo('Webcam reference not available');
      return;
    }
    
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) {
      setDebugInfo('Failed to get screenshot');
      return;
    }
    
    try {
      const imageData = screenshot.split(',')[1];
      
      // Always use the detect_emotion endpoint - it handles both recording and non-recording modes
      const response = await axios.post('http://localhost:8000/detect_emotion', 
        { image: imageData },
        { headers: { 'Content-Type': 'application/json' } }
      );
      
      // Only update UI with emotions when not in recording mode
      if (!isRecordingMode || !isRecording) {
        if (response.data && response.data.emotions) {
          setEmotions(response.data.emotions);
          setDebugInfo(`Detected ${response.data.emotions.length} faces`);
        } else {
          setEmotions([]);
          setDebugInfo('No emotions data in response');
        }
      } else {
        setDebugInfo('Recording frame...');
      }
    } catch (err) {
      console.error("Error capturing frame:", err);
      setDebugInfo(`Error: ${err.message}`);
    }
  };

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, []);

  // Define the green color for emotion boxes - using the EmotionWave green
  const greenBorderColor = "#7AB317";  // From --emotion-happy in index.css

  return (
    <Card title="Emotion Detection" variant="primary">
      <div className="space-y-6">
        <div className="webcam-container">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={640}
            height={480}
            className="webcam"
          />
          
          {isCapturing && (
            <div className="camera-status-indicator">
              {isRecording ? 'Recording' : 'Analyzing'}
            </div>
          )}
          
          {emotions && emotions.length > 0 && (
            <div className="emotion-overlay">
              {emotions.map((emotion, index) => {
                if (!emotion.face_location || emotion.face_location.length !== 4) {
                  console.warn('Invalid face_location data:', emotion);
                  return null;
                }
                
                const [x1, y1, x2, y2] = emotion.face_location;
                const leftPercent = (x1 / 640) * 100;
                const topPercent = (y1 / 480) * 100;
                const widthPercent = ((x2 - x1) / 640) * 100;
                const heightPercent = ((y2 - y1) / 480) * 100;
                
                return (
                  <div 
                    key={index} 
                    style={{
                      position: 'absolute',
                      left: `${leftPercent}%`,
                      top: `${topPercent}%`,
                      width: `${widthPercent}%`,
                      height: `${heightPercent}%`,
                      border: `3px solid ${greenBorderColor}`,
                      boxSizing: 'border-box',
                      zIndex: 10
                    }}
                  >
                    <div style={{
                      position: 'absolute',
                      top: '-28px',
                      left: '0',
                      background: greenBorderColor,
                      color: 'white',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '14px',
                      fontWeight: 'bold',
                      whiteSpace: 'nowrap'
                    }}>
                      {emotion.dominant_emotion}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
        
        {debugInfo && (
          <div className="debug-info" style={{
            marginBottom: '10px',
            fontSize: '12px',
            color: 'var(--neutral-dark)'
          }}>
            Debug: {debugInfo}
          </div>
        )}
        
        <div className="controls">
          <Button 
            onClick={handleCameraToggle}
            variant={isCapturing ? 'danger' : 'primary'}
            disabled={buttonLock}
          >
            {isCapturing ? (
              <>
                <i className="fas fa-stop mr-2"></i>
                Stop Camera
              </>
            ) : (
              <>
                <i className="fas fa-camera mr-2"></i>
                Start Camera
              </>
            )}
          </Button>
          
          <Button 
            onClick={handleRecordingToggle}
            variant={isRecording ? 'warning' : 'success'}
            disabled={!isCapturing || buttonLock}
          >
            {isRecording ? (
              <>
                <i className="fas fa-stop-circle mr-2"></i>
                Stop Recording
              </>
            ) : (
              <>
                <i className="fas fa-record-vinyl mr-2"></i>
                Start Recording
              </>
            )}
          </Button>
        </div>
        
        {buttonLock && (
          <div className="processing-indicator">
            <div className="spinner"></div>
            <p>Processing request...</p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default EmotionDetector;