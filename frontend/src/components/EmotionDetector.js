import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import Card from './Card';
import Button from './Button';

const EmotionDetector = ({ setAnalysisResults, isRecording, setIsRecording }) => {
  const webcamRef = useRef(null);
  const [emotions, setEmotions] = useState([]);
  const [buttonLock, setButtonLock] = useState(false);
  const captureIntervalRef = useRef(null);
  const [debugInfo, setDebugInfo] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageResults, setImageResults] = useState(null);
  const [isProcessingImage, setIsProcessingImage] = useState(false);  const [webcamActive, setWebcamActive] = useState(false); // Track webcam state explicitly
  const [hasCameraPermission, setHasCameraPermission] = useState(null); // Track camera permission status

  // Check for camera permissions on component mount
  useEffect(() => {
    const checkCameraPermission = async () => {
      try {
        // This will prompt for permission if not already granted
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        // If we get here, permission was granted
        setHasCameraPermission(true);
        // Stop the stream after getting permission since we'll start it again when needed
        stream.getTracks().forEach(track => track.stop());
      } catch (err) {
        console.error("Camera permission error:", err);
        setHasCameraPermission(false);
        setDebugInfo(`Camera permission denied: ${err.message}`);
      }
    };

    checkCameraPermission();
  }, []);

  // Capture a frame from the webcam
  const captureFrame = async () => {
    // Only capture if webcam is available
    if (!webcamRef.current || !webcamRef.current.video || webcamRef.current.video.readyState < 3) {
      // setDebugInfo('Capture skipped: Webcam not ready.');
      return;
    }

    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) {
      // setDebugInfo('Failed to get screenshot'); // Reduce noise
      return;
    }    const apiUrl = process.env.REACT_APP_API_URL; // Use environment variable without fallback   
    try {
      const imageData = screenshot.split(',')[1];
      // Use apiUrl here with /api/ prefix
      const response = await axios.post(`${apiUrl}/api/detect_emotion`,
        { image: imageData },
        { headers: { 'Content-Type': 'application/json' } }
      );

      // ALWAYS update live emotion boxes based on the response
      if (response.data && response.data.emotions) {
        setEmotions(response.data.emotions);
        // Optionally update debug info based on recording state if needed
        // setDebugInfo(isRecording ? 'Recording frame...' : `Detected ${response.data.emotions.length} faces`);
      } else {
        setEmotions([]);
        // setDebugInfo(isRecording ? 'Recording frame...' : 'No emotions data in response');
      }

    } catch (err) {
      console.error("Error capturing frame:", err);
      setDebugInfo(`Frame capture error: ${err.message}`);
    }
  };
  // Renamed and updated function to handle both camera and recording
  const handleDetectionToggle = async () => {
    setButtonLock(true);
    setDebugInfo('Processing request...'); // Give immediate feedback
    const apiUrl = process.env.REACT_APP_API_URL; // Use environment variable without fallback

    if (isRecording) {
      // Stop recording
      try {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;        // Use apiUrl here
        const response = await axios.post(`${apiUrl}/api/stop_recording/`);
        setAnalysisResults(response.data); // Update dashboard with results
        setIsRecording(false);
        setWebcamActive(false); // Deactivate webcam component
        setEmotions([]); // Clear emotion boxes
        setDebugInfo('Detection stopped.');

        // Explicitly stop the stream tracks
        if (webcamRef.current && webcamRef.current.stream) {
            const tracks = webcamRef.current.stream.getTracks();
            tracks.forEach(track => track.stop());
            webcamRef.current.stream = null; // Clear the stream reference
        }
         if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.srcObject) {
             const tracks = webcamRef.current.video.srcObject.getTracks();
             tracks.forEach(track => track.stop());
             webcamRef.current.video.srcObject = null; // Ensure video element stream is cleared
         }


      } catch (err) {
        console.error("Error stopping detection:", err);
        setDebugInfo(`Error stopping detection: ${err.response?.data?.detail || err.message}`);
        setTimeout(() => setButtonLock(false), 2000);
        return;
      }    } else {
      // Start recording
      try {
        // Check if we have camera permission
        if (hasCameraPermission === false) {
          // If permission was previously denied, try again
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            setHasCameraPermission(true);
          } catch (err) {
            console.error("Camera permission error:", err);
            setDebugInfo(`Cannot start detection: Camera access denied. Please enable camera access in your browser settings.`);
            setTimeout(() => setButtonLock(false), 2000);
            return;
          }
        }

        setWebcamActive(true);
        await new Promise(resolve => setTimeout(resolve, 100));

        // Use apiUrl here
        await axios.post(`${apiUrl}/api/start_recording/`);
        setIsRecording(true);

        // Start sending frames frequently
        captureIntervalRef.current = setInterval(() => {
          captureFrame(); // Call captureFrame without parameters
        }, 200); // Recording capture rate
        setDebugInfo('Detection started.');

      } catch (err) {
        console.error("Error starting detection:", err);
        setDebugInfo(`Error starting detection: ${err.response?.data?.detail || err.message}`);
        setIsRecording(false);
        setWebcamActive(false);
        if (captureIntervalRef.current) {
          clearInterval(captureIntervalRef.current);
          captureIntervalRef.current = null;
        }
        setTimeout(() => setButtonLock(false), 2000);
        return;
      }
    }

    setTimeout(() => setButtonLock(false), 1000);
  };

  // Handle image uploads (no changes needed here)
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedImage(URL.createObjectURL(file));
    setImageResults(null); // Clear previous image results
    setIsProcessingImage(true);
    setDebugInfo('Processing uploaded image...');
    const apiUrl = process.env.REACT_APP_API_URL;

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Use apiUrl here
      const response = await axios.post(
        `${apiUrl}/api/detect_emotion_from_image`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (response.data && response.data.emotions) {
        setImageResults(response.data.emotions);
        setDebugInfo(`Detected ${response.data.emotions.length} faces in uploaded image`);
      } else {
        setImageResults([]);
        setDebugInfo('No emotions detected in image');
      }
    } catch (err) {
      console.error("Error processing image:", err);
      setDebugInfo(`Image processing error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsProcessingImage(false);
    }
  };

  // Clean up interval and stop webcam on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      // Stop webcam stream on unmount
       if (webcamRef.current && webcamRef.current.stream) {
           const tracks = webcamRef.current.stream.getTracks();
           tracks.forEach(track => track.stop());
       }
        if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.srcObject) {
             const tracks = webcamRef.current.video.srcObject.getTracks();
             tracks.forEach(track => track.stop());
             webcamRef.current.video.srcObject = null;
         }
      // Optional: Stop backend recording if unmounting while recording is active
      // Consider if this is desired behavior as it might lead to incomplete recordings.
      // if (isRecording) {
      //   axios.post('http://localhost:8000/stop_recording/').catch(err => console.error("Error stopping recording on unmount:", err));
      // }
    };
  }, []); // Run cleanup only once on unmount

  return (
    <Card title="Emotion Detector" variant="dark">
      <div className="webcam-container" style={{ position: 'relative', minHeight: '300px', background: '#333', borderRadius: 'var(--border-radius)' }}>
        {webcamActive ? (
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={{ width: 640, height: 480 }}
            className="webcam" // Use class for styling
            mirrored={true}
            onUserMedia={() => setDebugInfo('Webcam ready.')}
            onUserMediaError={(err) => setDebugInfo(`Webcam Error: ${err.name}`)}
          />        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: '300px', color: 'var(--neutral-light)' }}>
            {hasCameraPermission === false ? 
              'Camera permission denied. Please enable camera access in your browser settings.' : 
              'Camera is off'}
          </div>
        )}

        {/* Draw emotion boxes on top of the webcam feed */}
        {webcamActive && emotions && emotions.length > 0 && (
          <div className="emotion-overlay" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            {emotions.map((emotion, index) => {
              // Adjust coordinates based on mirrored view if necessary
              // Assuming 640x480 webcam resolution
              const videoWidth = 640;
              const videoHeight = 480;
              const [x1_orig, y1, x2_orig, y2] = emotion.face_location;

              // Mirror the x-coordinates
              const x1 = videoWidth - x2_orig;
              const x2 = videoWidth - x1_orig;

              const leftPercent = (x1 / videoWidth) * 100;
              const topPercent = (y1 / videoHeight) * 100;
              const widthPercent = ((x2 - x1) / videoWidth) * 100;
              const heightPercent = ((y2 - y1) / videoHeight) * 100;

              return (
                <div
                  key={index}
                  style={{
                    position: 'absolute',
                    left: `${leftPercent}%`,
                    top: `${topPercent}%`,
                    width: `${widthPercent}%`,
                    height: `${heightPercent}%`,
                    border: `3px solid rgba(46, 204, 113, 0.8)`, // Green for detection boxes
                    boxSizing: 'border-box',
                    zIndex: 10
                  }}
                >
                  <div style={{
                    position: 'absolute',
                    top: '-28px', // Position label above the box
                    left: '0',
                    background: 'rgba(46, 204, 113, 0.8)',
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
          marginTop: '10px', // Add margin top
          marginBottom: '10px',
          fontSize: '12px',
          color: 'var(--neutral-light)', // Lighter color for dark card
          textAlign: 'center'
        }}>
          Status: {debugInfo}
        </div>
      )}

      <div className="controls" style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        {/* Combined Start/Stop Button */}
        <Button
          onClick={handleDetectionToggle}
          variant={isRecording ? 'danger' : 'success'} // Red when recording, Green when stopped
          disabled={buttonLock}
        >
          {isRecording ? (
            <>
              <i className="fas fa-stop-circle mr-2"></i>
              Stop Detection
            </>
          ) : (
            <>
              <i className="fas fa-play-circle mr-2"></i>
              Start Detection
            </>
          )}
        </Button>
      </div>

      {/* Keep button lock indicator */}
      {buttonLock && (
        <div className="processing-indicator">
          {/* Spinner or loading text can go here if needed, but debugInfo provides status */}
        </div>
      )}

      {/* Image Upload Section */}
      <div className="image-upload-section" style={{ marginTop: '2rem', borderTop: '1px solid var(--neutral-dark)', paddingTop: '1.5rem' }}>
        <h3 style={{ textAlign: 'center', marginBottom: '1rem', color: 'var(--neutral-lightest)' }}>
          <i className="fas fa-cloud-upload-alt mr-2"></i>
          Upload Image for Emotion Detection
        </h3>

        <div className="custom-file-upload" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <input
            type="file"
            id="file-upload"
            accept="image/*"
            onChange={handleImageUpload}
            disabled={isProcessingImage || isRecording} // Also disable if recording
            className="file-input-hidden" // Keep hidden
          />
          <label htmlFor="file-upload" className={`file-upload-label ${isProcessingImage || isRecording ? 'disabled' : ''}`}>
            <i className="fas fa-image mr-2"></i>
            {isProcessingImage ? 'Processing...' : 'Choose an image'}
          </label>

          {selectedImage && !isProcessingImage && (
            <div className="selected-file-name" style={{ fontSize: '0.9em', color: 'var(--neutral-light)' }}>
              {/* Display filename or confirmation */}
              {/* {selectedImage.substring(selectedImage.lastIndexOf('/') + 1)} */}
            </div>
          )}
        </div>

        {selectedImage && (
          <>
            {/* Container for the image */}
            <div className="uploaded-image-container">
              <img
                src={selectedImage}
                alt="Uploaded"
              />
            </div>

            {/* Display analysis results BELOW the image */}
            {imageResults && imageResults.length > 0 && (
              <div className="image-analysis-details">
                <h4>Analysis Results</h4>
                {imageResults.map((result, index) => (
                  <div key={index} className="face-result-item">
                    <p><strong>Face {index + 1}:</strong> Dominant Emotion - <strong>{result.dominant_emotion}</strong></p>
                    {result.emotion_scores && (
                      <>
                        <p>Emotion Scores:</p>
                        <ul>
                          {Object.entries(result.emotion_scores)
                            .sort(([, scoreA], [, scoreB]) => scoreB - scoreA) // Sort by score descending
                            .map(([emotion, score]) => (
                              <li key={emotion}>
                                {emotion.charAt(0).toUpperCase() + emotion.slice(1)}: {score.toFixed(1)}%
                              </li>
                            ))}
                        </ul>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Handle case where image is selected but no faces detected */}
            {imageResults && imageResults.length === 0 && (
               <div className="image-analysis-details">
                 <h4>Analysis Results</h4>
                 <div className="face-result-item">
                   <p>No faces were detected in the uploaded image.</p>
                 </div>
               </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
};

export default EmotionDetector;