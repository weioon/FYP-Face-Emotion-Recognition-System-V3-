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

  const chartData = {
    labels: Object.keys(analysisResults.stats || {}),
    datasets: [
      {
        label: 'Emotion Distribution (%)',
        data: Object.values(analysisResults.stats || {}),
        backgroundColor: [
          'rgba(241, 196, 15, 0.7)',   // Happy - Yellow
          'rgba(52, 152, 219, 0.7)',   // Sad - Blue
          'rgba(231, 76, 60, 0.7)',    // Angry - Red
          'rgba(155, 89, 182, 0.7)',   // Surprise - Purple
          'rgba(26, 188, 156, 0.7)',  // Fear - Teal
          'rgba(22, 160, 133, 0.7)',  // Disgust - Dark Teal
          'rgba(149, 165, 166, 0.7)'  // Neutral - Gray
        ],
        borderColor: [
          'rgba(241, 196, 15, 1)',
          'rgba(52, 152, 219, 1)',
          'rgba(231, 76, 60, 1)',
          'rgba(155, 89, 182, 1)',
          'rgba(26, 188, 156, 1)',
          'rgba(22, 160, 133, 1)',
          'rgba(149, 165, 166, 1)'
        ],
        borderWidth: 1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Emotion Distribution',
        font: {
          family: 'Playfair Display',
          size: 16
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100
      }
    }
  };

  return (
    <div className="emotion-dashboard space-y-6">
      <Card title="Emotion Analysis Results" variant="primary">
        <div className="analysis-section">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2">Session Overview</h3>
              <p className="text-gray-700">
                <span className="font-medium">Duration:</span> {analysisResults.duration?.toFixed(2)} seconds
              </p>
              <p className="text-gray-700">
                <span className="font-medium">Dominant Emotion:</span> {analysisResults.dominant_emotion}
              </p>
            </div>
            
            <div className="chart-container">
              <Bar data={chartData} options={chartOptions} />
            </div>
          </div>
        </div>
      </Card>

      {analysisResults.emotion_journey && (
        <Card title="Emotional Journey" variant="success">
          <div className="analysis-section">
            <div className="journey-stages grid grid-cols-1 md:grid-cols-3 gap-4">
              {['beginning', 'middle', 'end'].map((stage, index) => (
                <div key={stage} className={`stage p-4 rounded-lg ${index === 0 ? 'bg-blue-50' : index === 1 ? 'bg-purple-50' : 'bg-teal-50'}`}>
                  <h4 className="text-lg font-semibold mb-3 text-center capitalize">{stage}</h4>
                  <ul className="space-y-2">
                    {Object.entries(analysisResults.emotion_journey[stage] || {}).map(([emotion, value]) => (
                      <li key={emotion} className="flex justify-between items-center bg-white p-2 rounded">
                        <span className="capitalize">{emotion}</span>
                        <span className="font-medium">{typeof value === 'number' ? value.toFixed(1) : value}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {analysisResults.interpretation && (
        <Card title="Interpretation" variant="warning">
          <div className="analysis-section">
            <div className="prose max-w-none">
              <p className="whitespace-pre-line">{analysisResults.interpretation}</p>
            </div>
          </div>
        </Card>
      )}

      {analysisResults.educational_recommendations && (
        <Card title="Educational Recommendations">
          <div className="analysis-section">
            <ul className="space-y-3">
              {analysisResults.educational_recommendations.map((rec, index) => (
                <li key={index} className="flex items-start">
                  <span className="inline-block bg-primary-color text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 mt-1 flex-shrink-0">
                    {index + 1}
                  </span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      )}
    </div>
  );
};

export default EmotionDashboard;