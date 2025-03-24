import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const EmotionDetector = ({ setAnalysisResults, isRecording, setIsRecording }) => {
  const webcamRef = useRef(null);
  const [emotions, setEmotions] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [buttonLock, setButtonLock] = useState(false);
  const captureIntervalRef = useRef(null);
  const [webcamDimensions, setWebcamDimensions] = useState({ width: 640, height: 480 });
  const webcamContainerRef = useRef(null);

  // Add authentication headers to axios requests
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // Apply token to every request
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      console.log("Added auth token to requests");
    } else {
      console.error("No authentication token found in localStorage");
    }
  }, []);

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
    
    try {
      if (isCapturing) {
        stopCapturing();
      } else {
        startCapturing();
      }
    } catch (error) {
      console.error("Camera toggle error:", error);
    } finally {
      // Always release the button lock
      setTimeout(() => setButtonLock(false), 1000);
    }
  };

  // Update the handleRecordingToggle function
  const handleRecordingToggle = async () => {
    if (!isCapturing) return; // Don't proceed if camera isn't on
    
    // Set button lock
    setButtonLock(true);
    
    try {
      if (!isRecording) {
        console.log("Starting recording session...");
        
        try {
          const response = await axios.post('http://localhost:8000/start_recording/', {}, { 
            timeout: 5000  // Increased timeout 
          });
          console.log("Recording started response:", response.data);
          setIsRecording(true);
        } catch (err) {
          console.error("Recording start error:", err);
          throw err; // Re-throw to be caught by outer catch
        }
      } else {
        console.log("Stopping recording session...");
        try {
          const response = await axios.post('http://localhost:8000/stop_recording/', {}, { 
            timeout: 8000  // Increased timeout for analysis
          });
          console.log("Recording stopped with results:", response.data);
          setIsRecording(false);
          setAnalysisResults(response.data);
        } catch (err) {
          console.error("Recording stop error:", err);
          setIsRecording(false);
          throw err; // Re-throw to be caught by outer catch
        }
      }
    } catch (error) {
      console.error("Recording toggle error:", error);
      // Don't show alert, just log the error
      setIsRecording(false);
    } finally {
      // Always release the button lock with a delay
      setTimeout(() => setButtonLock(false), 500);
    }
  };

  // Replace your frame capture function with this more robust version
  const captureFrame = async () => {
    if (!webcamRef.current) return;
    
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;
      
      const base64Image = imageSrc.split(',')[1];
      
      // Use shorter timeout (2s) to prevent UI blocking
      const response = await axios.post(
        'http://localhost:8000/detect_emotion', 
        { image: base64Image },
        { 
          timeout: 2000,
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }
      );
      
      if (response.data && response.data.emotions) {
        setEmotions(response.data.emotions);
      }
    } catch (error) {
      console.error("Frame capture error:", error);
      // Don't update state on error
    }
  };

  // Capture frame function - run when camera is active
  useEffect(() => {
    if (!isCapturing) return;
    
    console.log("Setting up frame capture...");
    
    // Initial capture with a delay to allow webcam to initialize
    setTimeout(captureFrame, 1000);
    
    // Use a slower capture rate (every 1.5 seconds instead of every 500ms)
    captureIntervalRef.current = setInterval(captureFrame, 1500);
    
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

  // Update the webcam dimensions effect
  useEffect(() => {
    if (!isCapturing || !webcamRef.current) return;
    
    const updateDimensions = () => {
      if (webcamRef.current && webcamRef.current.video) {
        const video = webcamRef.current.video;
        
        // Log actual dimensions for debugging
        console.log("Webcam dimensions:", {
          width: video.videoWidth,
          height: video.videoHeight,
          displayWidth: video.clientWidth,
          displayHeight: video.clientHeight
        });
        
        setWebcamDimensions({
          width: video.clientWidth,
          height: video.clientHeight,
          videoWidth: video.videoWidth || 640,
          videoHeight: video.videoHeight || 480
        });
      }
    };
    
    // Initial update with delay
    const initialUpdate = setTimeout(updateDimensions, 1000);
    
    // Update on resize
    window.addEventListener('resize', updateDimensions);
    
    return () => {
      clearTimeout(initialUpdate);
      window.removeEventListener('resize', updateDimensions);
    };
  }, [isCapturing, webcamRef.current]);

  // Add this CSS with the appropriate selectors
  const overlayStyles = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none'
  };

  return (
    <div className="emotion-detector">
      <div className="webcam-container" ref={webcamContainerRef}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={640}
          height={480}
          className="webcam"
        />
        
        {emotions.length > 0 && (
          <div className="emotion-overlay" style={overlayStyles}>
            {emotions.map((emotion, index) => {
              // Get exact webcam element dimensions
              const webcamElement = webcamRef.current.video;
              const displayWidth = webcamElement.clientWidth;
              const displayHeight = webcamElement.clientHeight;
              
              // Extract face coordinates - Handle different formats
              let x1, y1, x2, y2;
              if (emotion.face_location.length === 4) {
                [x1, y1, x2, y2] = emotion.face_location;
              } else {
                x1 = emotion.face_location[0];
                y1 = emotion.face_location[1];
                x2 = x1 + (emotion.face_location[2] || 100);
                y2 = y1 + (emotion.face_location[3] || 100);
              }
              
              // Calculate box dimensions as percentages (more reliable across different sizes)
              const leftPercent = (x1 / 640) * 100;
              const topPercent = (y1 / 480) * 100;
              const widthPercent = ((x2 - x1) / 640) * 100;
              const heightPercent = ((y2 - y1) / 480) * 100;
              
              return (
                <div 
                  key={index} 
                  className="emotion-box"
                  style={{
                    position: 'absolute',
                    left: `${leftPercent}%`,
                    top: `${topPercent}%`,
                    width: `${widthPercent}%`,
                    height: `${heightPercent}%`,
                    border: '2px solid green',
                    boxSizing: 'border-box'
                  }}
                >
                  <div className="emotion-label"
                    style={{
                      position: 'absolute',
                      top: '-25px',
                      left: '0',
                      backgroundColor: 'green',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '14px',
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