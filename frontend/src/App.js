import React, { useState } from 'react';
import './App.css';
import EmotionDetector from './components/EmotionDetector';
import EmotionDashboard from './components/EmotionDashboard';
import Header from './components/Header';

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  
  return (
    <div className="App">
      <Header />
      <div className="container">
        <div className="row">
          <div className="col-md-7">
            <EmotionDetector 
              setAnalysisResults={setAnalysisResults}
              isRecording={isRecording}
              setIsRecording={setIsRecording}
            />
          </div>
          <div className="col-md-5">
            <EmotionDashboard 
              analysisResults={analysisResults}
              isRecording={isRecording}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;