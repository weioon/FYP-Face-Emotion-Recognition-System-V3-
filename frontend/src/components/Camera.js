import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';

const Camera = () => {
  const videoRef = useRef(null);
  const [emotionData, setEmotionData] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    let interval;
    
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    if (isStreaming) {
      startCamera();
      interval = setInterval(() => {
        captureAndSendFrame();
      }, 1000); // Send frame every second
    } else if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }

    return () => {
      if (interval) clearInterval(interval);
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, [isStreaming]);

  const captureAndSendFrame = () => {
    if (!videoRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];
    
    // Send to backend
    axios.post('http://localhost:8000/detect_emotion', { 
      image: base64Image 
    })
      .then(response => {
        setEmotionData(response.data);
      })
      .catch(error => {
        console.error("Error detecting emotion:", error);
      });
  };

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
      />
      
      <div className="controls">
        <button 
          onClick={() => setIsStreaming(!isStreaming)}
          className="control-button"
        >
          {isStreaming ? 'Stop Camera' : 'Start Camera'}
        </button>
      </div>

      {emotionData && (
        <div className="results">
          <h3>Detected Emotions:</h3>
          <pre>{JSON.stringify(emotionData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default Camera;