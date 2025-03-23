import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';

const WebcamCapture = () => {
  const videoRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Get webcam access when component mounts
    async function setupWebcam() {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ 
          video: true, 
          audio: false 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          setStream(mediaStream);
          setError(null);
        }
      } catch (err) {
        setError("Error accessing webcam: " + err.message);
        console.error("Webcam access error:", err);
      }
    }
    
    setupWebcam();
    
    // Cleanup on component unmount
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      await axios.post('http://localhost:8000/start_recording/');
      setIsRecording(true);
      captureFrames();
    } catch (err) {
      console.error("Error starting recording:", err);
      setError("Failed to start recording");
    }
  };

  const stopRecording = async () => {
    try {
      setIsRecording(false);
      const response = await axios.post('http://localhost:8000/stop_recording/');
      setResult(response.data);
    } catch (err) {
      console.error("Error stopping recording:", err);
      setError("Failed to stop recording");
    }
  };

  const captureFrames = () => {
    if (!isRecording) return;
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const video = videoRef.current;
    
    if (video) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(async (blob) => {
        if (blob && isRecording) {
          try {
            await axios.post('http://localhost:8000/process_frame/', blob, {
              headers: {
                'Content-Type': 'application/octet-stream'
              }
            });
            setTimeout(captureFrames, 100); // Send frame every 100ms
          } catch (err) {
            console.error("Error sending frame:", err);
          }
        }
      }, 'image/jpeg');
    }
  };

  return (
    <div className="webcam-container">
      {error && <div className="error-message">{error}</div>}
      
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: '100%', maxWidth: '640px' }}
      />
      
      <div className="controls">
        {!isRecording ? (
          <button onClick={startRecording} className="record-btn">
            Start Recording
          </button>
        ) : (
          <button onClick={stopRecording} className="stop-btn">
            Stop Recording
          </button>
        )}
      </div>
      
      {result && (
        <div className="result">
          <h3>Analysis Results</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;