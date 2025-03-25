import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const EmotionDashboard = ({ analysisResults, isRecording }) => {
  if (isRecording) {
    return (
      <div className="emotion-dashboard">
        <div className="recording-message">
          <h2>Recording in Progress</h2>
          <p>Capturing emotional data for analysis...</p>
          <div className="recording-indicator"></div>
        </div>
      </div>
    );
  }

  if (!analysisResults) {
    return (
      <div className="emotion-dashboard">
        <div className="no-data-message">
          <h2>No Analysis Data</h2>
          <p>Start a recording session to see emotion analysis results.</p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const emotionLabels = Object.keys(analysisResults.stats || {});
  const emotionValues = Object.values(analysisResults.stats || {});
  
  const chartData = {
    labels: emotionLabels,
    datasets: [
      {
        label: 'Emotion Distribution (%)',
        data: emotionValues,
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',   // Red - Angry
          'rgba(54, 162, 235, 0.6)',   // Blue - Sad
          'rgba(255, 206, 86, 0.6)',   // Yellow - Happy
          'rgba(75, 192, 192, 0.6)',   // Green - Surprise
          'rgba(153, 102, 255, 0.6)',  // Purple - Fear
          'rgba(255, 159, 64, 0.6)',   // Orange - Disgust
          'rgba(201, 203, 207, 0.6)'   // Grey - Neutral
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(54, 162, 235)',
          'rgb(255, 206, 86)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
          'rgb(255, 159, 64)',
          'rgb(201, 203, 207)'
        ],
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="emotion-dashboard">
      <h2>Emotion Analysis Results</h2>
      <p>Recording Duration: {analysisResults.duration?.toFixed(2)} seconds</p>
      
      <div className="chart-container">
        <h3>Emotion Distribution</h3>
        <Bar data={chartData} />
      </div>
      
      {analysisResults.emotion_journey && (
        <div className="journey-analysis">
          <h3>Emotional Journey</h3>
          <div className="journey-stages">
            <div className="stage">
              <h4>Beginning</h4>
              <ul>
                {Object.entries(analysisResults.emotion_journey.beginning || {}).map(([emotion, value]) => (
                  <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                ))}
              </ul>
            </div>
            <div className="stage">
              <h4>Middle</h4>
              <ul>
                {Object.entries(analysisResults.emotion_journey.middle || {}).map(([emotion, value]) => (
                  <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                ))}
              </ul>
            </div>
            <div className="stage">
              <h4>End</h4>
              <ul>
                {Object.entries(analysisResults.emotion_journey.end || {}).map(([emotion, value]) => (
                  <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
      
      {analysisResults.interpretation && (
        <div className="interpretation">
          <h3>Interpretation</h3>
          <p style={{ whiteSpace: 'pre-line' }}>{analysisResults.interpretation}</p>
        </div>
      )}
      
      {analysisResults.significant_shifts && (
        <div className="shifts">
          <h3>Significant Emotional Shifts</h3>
          {analysisResults.significant_shifts.length > 0 ? (
            <ul>
              {analysisResults.significant_shifts.map((shift, index) => (
                <li key={index}>{shift}</li>
              ))}
            </ul>
          ) : (
            <p>No significant emotional shifts detected</p>
          )}
        </div>
      )}
      
      {analysisResults.educational_recommendations && (
        <div className="recommendations">
          <h3>Educational Recommendations</h3>
          <ul>
            {analysisResults.educational_recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default EmotionDashboard;