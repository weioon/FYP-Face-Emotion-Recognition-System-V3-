import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const EmotionDetector = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [processing, setProcessing] = useState(false);

  const processFrame = async () => {
    if (!webcamRef.current || processing) return;
    setProcessing(true);

    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      try {
        // Convert to blob
        const blob = await fetch(imageSrc).then((res) => res.blob());
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        // Send to backend
        const response = await axios.post('http://localhost:8000/detect', formData);
        const { image } = response.data;

        // Update canvas
        const img = new Image();
        img.src = `data:image/jpeg;base64,${image}`;
        img.onload = () => {
          const canvas = canvasRef.current;
          canvas.getContext('2d').drawImage(img, 0, 0);
        };
      } catch (error) {
        console.error('Error processing frame:', error);
      }
    }
    setProcessing(false);
  };

  useEffect(() => {
    const interval = setInterval(processFrame, 300); // Adjust interval
    return () => clearInterval(interval);
  }, [processing]);

  return (
    <div style={{ position: 'relative' }}>
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={640}
        height={480}
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ position: 'absolute', left: 0, top: 0 }}
      />
    </div>
  );
};

export default EmotionDetector;