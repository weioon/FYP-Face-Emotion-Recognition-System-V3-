import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import Card from './Card';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const EmotionDashboard = ({ analysisResults, isRecording }) => {
  if (isRecording) {
    return (
      <Card title="Recording Session" variant="primary">
        <div className="recording-message">
          <h3 className="text-center text-xl font-semibold mb-2">Recording in Progress</h3>
          <p className="text-center text-gray-600">Capturing emotional data for analysis...</p>
          <div className="recording-indicator"></div>
        </div>
      </Card>
    );
  }

  if (!analysisResults) {
    return (
      <Card title="Emotion Analysis" variant="default">
        <div className="no-data-message text-center py-8">
          <h3 className="text-xl font-semibold mb-2">No Analysis Data</h3>
          <p className="text-gray-600">Start a recording session to see emotion analysis results.</p>
        </div>
      </Card>
    );
  }

  // Handle multi-face results
  if (analysisResults.faces && analysisResults.faces.length > 0) {
    return (
      <div className="emotion-dashboard space-y-6">
        <h2 className="text-xl font-bold mb-4">
          Analysis Results ({analysisResults.face_count} {analysisResults.face_count === 1 ? 'person' : 'people'} detected)
        </h2>
        
        {analysisResults.faces.map((faceData, index) => {
          // Prepare chart data for this face
          const chartData = {
            labels: Object.keys(faceData.stats || {}),
            datasets: [
              {
                label: 'Emotion Distribution (%)',
                data: Object.values(faceData.stats || {}),
                backgroundColor: [
                  'rgba(241, 196, 15, 0.7)',   // Happy - Yellow
                  'rgba(52, 152, 219, 0.7)',   // Sad - Blue
                  'rgba(231, 76, 60, 0.7)',    // Angry - Red
                  'rgba(155, 89, 182, 0.7)',   // Surprise - Purple
                  'rgba(26, 188, 156, 0.7)',  // Fear - Teal
                  'rgba(22, 160, 133, 0.7)',  // Disgust - Dark Teal
                  'rgba(149, 165, 166, 0.7)',  // Neutral - Gray
                ]
              }
            ]
          };

          const chartOptions = {
            plugins: {
              legend: {
                position: 'bottom',
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                max: 100
              }
            }
          };
          
          return (
            <Card key={index} title={`Person ${index+1} (ID:${faceData.face_id})`} variant="primary">
              <div className="analysis-section">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold mb-2">Session Overview</h3>
                    <p className="text-gray-700">
                      <span className="font-medium">Duration:</span> {faceData.duration?.toFixed(2)} seconds
                    </p>
                    <p className="text-gray-700">
                      <span className="font-medium">Dominant Emotion:</span> {faceData.dominant_emotion}
                    </p>
                  </div>
                  
                  <div className="chart-container">
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                </div>
                
                {/* Emotional journey section */}
                {faceData.emotion_journey && (
                  <div className="mt-6">
                    <h3 className="font-semibold mb-2">Emotional Journey</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="journey-phase">
                        <h4 className="text-sm font-medium">Beginning</h4>
                        <ul className="text-sm">
                          {Object.entries(faceData.emotion_journey.beginning).map(([emotion, value]) => (
                            <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                          ))}
                        </ul>
                      </div>
                      <div className="journey-phase">
                        <h4 className="text-sm font-medium">Middle</h4>
                        <ul className="text-sm">
                          {Object.entries(faceData.emotion_journey.middle).map(([emotion, value]) => (
                            <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                          ))}
                        </ul>
                      </div>
                      <div className="journey-phase">
                        <h4 className="text-sm font-medium">End</h4>
                        <ul className="text-sm">
                          {Object.entries(faceData.emotion_journey.end).map(([emotion, value]) => (
                            <li key={emotion}>{emotion}: {value.toFixed(1)}%</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Interpretation and recommendations */}
                {faceData.interpretation && (
                  <div className="mt-6">
                    <h3 className="font-semibold mb-2">Analysis</h3>
                    <p className="text-gray-700">{faceData.interpretation}</p>
                  </div>
                )}
                
                {faceData.educational_recommendations && (
                  <div className="mt-4">
                    <h3 className="font-semibold mb-2">Recommendations</h3>
                    <ul className="list-disc pl-5 text-gray-700">
                      {faceData.educational_recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    );
  }
  
  // Fallback to original single-face display if the new format isn't available
  // [Keep your existing single-face code here]
  
  // ... rest of your existing component code
};

export default EmotionDashboard;