import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import moment from 'moment';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import Card from './Card';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const RecordingDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [recording, setRecording] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    // Existing effect code to fetch recording details
    const fetchRecording = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          navigate('/login');
          return;
        }

        const response = await axios.get(`http://localhost:8000/recording/${id}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        console.log("Recording data:", response.data);
        setRecording(response.data);
        setLoading(false);
      } catch (error) {
        setError('Failed to fetch recording details');
        setLoading(false);
        console.error('Error fetching recording:', error);
      }
    };

    fetchRecording();
  }, [id, navigate]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;
  if (!recording) return <div className="not-found">Recording not found</div>;

  // Helper function to determine if this is a multi-face recording
  const isMultiFaceRecording = () => {
    return recording.analysis_data.faces && recording.analysis_data.faces.length > 0;
  };

  // Updated getSummaryData function
  const getSummaryData = () => {
    if (isMultiFaceRecording()) {
      const faceCount = recording.analysis_data.faces.length;
      let summary = `This recording captured ${faceCount} ${faceCount === 1 ? 'person' : 'people'}.`;
      
      // Add summary for each face
      recording.analysis_data.faces.forEach((face, index) => {
        summary += ` Person ${index+1} showed primarily ${face.dominant_emotion}`;
        
        if (face.significant_emotions && face.significant_emotions.length > 1) {
          summary += ` (${face.significant_emotions[0].percentage.toFixed(1)}%), followed by ${face.significant_emotions[1].emotion} (${face.significant_emotions[1].percentage.toFixed(1)}%).`;
        } else {
          summary += '.';
        }
      });
      
      return summary;
    }
    
    // Fall back to the original logic for old recordings
    if (recording.analysis_data.summary) {
      return recording.analysis_data.summary;
    }
    
    if (recording.analysis_data.dominant_emotion) {
      return `The dominant emotion detected was ${recording.analysis_data.dominant_emotion}`;
    }
    
    if (recording.analysis_data.significant_emotions && 
        recording.analysis_data.significant_emotions.length > 0) {
      
      const emotions = recording.analysis_data.significant_emotions;
      let summary = `The primary emotion detected was ${emotions[0].emotion} (${emotions[0].percentage.toFixed(1)}%)`;
      
      if (emotions.length > 1) {
        summary += `, followed by ${emotions[1].emotion} (${emotions[1].percentage.toFixed(1)}%)`;
      }
      
      if (recording.analysis_data.emotion_stability !== undefined) {
        const stability = recording.analysis_data.emotion_stability > 0.7 ? 'stable' : 
                        (recording.analysis_data.emotion_stability > 0.4 ? 'moderately changing' : 'highly variable');
        summary += `. The emotional pattern was ${stability} throughout the session.`;
      }
      
      return summary;
    }
    
    return "Summary analysis could not be generated due to insufficient emotion data.";
  };

  const summaryData = getSummaryData();

  // --- Use the EXACT SAME chart options as EmotionDashboard.js ---
  const chartOptions = {
    responsive: true, // Let the chart resize within its container
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          // Ensure color matches the dark card background
          color: 'var(--neutral-lightest)', // Or '#FFFFFF'
        }
      }
      // No complex titles or specific axis settings needed if
      // EmotionDashboard doesn't have them. Keep it simple.
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          color: 'var(--neutral-lightest)', // Or '#FFFFFF'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)' // Light grid lines for dark bg
        }
      },
      x: {
        ticks: {
          color: 'var(--neutral-lightest)', // Or '#FFFFFF'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)' // Light grid lines for dark bg
        }
      }
    }
    // Ensure maintainAspectRatio is NOT set to false, or remove it entirely
    // maintainAspectRatio: true, // (Default) or remove this line
  };
  // --- End of chart options ---

  return (
    <div className="emotion-dashboard space-y-6">
      {/* Recording metadata */}
      <Card title="Recording Details" variant="primary">
        <div className="recording-meta">
          <p><i className="far fa-calendar-alt"></i> Date: {moment.utc(recording.timestamp).local().format('DD-MM-YYYY')}</p>
          <p><i className="far fa-clock"></i> Time: {moment.utc(recording.timestamp).local().format('h:mm:ss A')}</p>
          <p><i className="fas fa-stopwatch"></i> Duration: {recording.analysis_data.duration ? `${Math.round(recording.analysis_data.duration)} seconds` : 'N/A'}</p>
          {isMultiFaceRecording() && (
            <p><i className="fas fa-users"></i> People Detected: {recording.analysis_data.face_count}</p>
          )}
        </div>
      </Card>

      {/* Summary section */}
      <Card title="Analysis Summary" variant="primary">
        <div className="py-4">
          <p className="text-neutral-lightest">{summaryData}</p>
        </div>
      </Card>

      {/* Multi-face section */}
      {isMultiFaceRecording() ? (
        <>
          {recording.analysis_data.faces.map((faceData, index) => {
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
                    'rgba(26, 188, 156, 0.7)',   // Fear - Teal
                    'rgba(22, 160, 133, 0.7)',   // Disgust - Dark Teal
                    'rgba(149, 165, 166, 0.7)',  // Neutral - Gray
                  ]
                }
              ]
            };
            
            return (
              <Card key={index} title={`Person ${index+1} (ID:${faceData.face_id})`} variant="primary">
                <div className="analysis-section">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      {/* Title: White */}
                      <h3 className="font-semibold mb-2 text-white">Session Overview</h3>
                      {/* Content: Ensure content text is light on dark background */}
                      <p className="text-neutral-lightest">
                        <span className="font-medium">Duration:</span> {faceData.duration?.toFixed(2)} seconds
                      </p>
                      <p className="text-neutral-lightest">
                        <span className="font-medium">Dominant Emotion:</span> {faceData.dominant_emotion}
                      </p>
                    </div>

                    {/* Chart container remains unchanged */}
                    <div className="chart-container">
                      <Bar data={chartData} options={chartOptions} />
                    </div>
                  </div>

                  {/* Emotional journey section */}
                  {faceData.emotion_journey && (
                    <div className="mt-6">
                      {/* Title: White */}
                      <h3 className="font-semibold mb-2 text-white">Emotional Journey</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          {/* Sub-title: White */}
                          <h4 className="text-sm font-medium text-white">Beginning</h4>
                          <ul className="text-sm">
                            {/* Content: Light */}
                            {Object.entries(faceData.emotion_journey.beginning || {}).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                            ))}
                          </ul>
                        </div>
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          {/* Sub-title: White */}
                          <h4 className="text-sm font-medium text-white">Middle</h4>
                          <ul className="text-sm">
                            {/* Content: Light */}
                            {Object.entries(faceData.emotion_journey.middle || {}).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                            ))}
                          </ul>
                        </div>
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          {/* Sub-title: White */}
                          <h4 className="text-sm font-medium text-white">End</h4>
                          <ul className="text-sm">
                            {/* Content: Light */}
                            {Object.entries(faceData.emotion_journey.end || {}).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Interpretation and recommendations */}
                  {faceData.interpretation && (
                    <div className="mt-6">
                      {/* Title: White */}
                      <h3 className="font-semibold mb-2 text-white">Analysis</h3>
                      {/* Content: Light */}
                      <p className="text-neutral-lightest">{faceData.interpretation}</p>
                    </div>
                  )}

                  {faceData.educational_recommendations && (
                    <div className="mt-4">
                      {/* Title: White */}
                      <h3 className="font-semibold mb-2 text-white">Recommendations</h3>
                      <ul className="list-disc pl-5">
                        {/* Content: Light */}
                        {faceData.educational_recommendations.map((rec, i) => (
                          <li key={i} className="text-neutral-lightest">{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </>
      ) : (
        // Legacy single-face display
        <Card title="Emotion Analysis" variant="primary">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              {/* Title: White */}
              <h3 className="font-semibold mb-2 text-white">Session Overview</h3>
              {/* Content: Light */}
              <p className="text-neutral-lightest">
                <span className="font-medium">Duration:</span> {recording.analysis_data.duration?.toFixed(2)} seconds
              </p>
              <p className="text-neutral-lightest">
                <span className="font-medium">Dominant Emotion:</span> {recording.analysis_data.dominant_emotion ||
                  (recording.analysis_data.significant_emotions && recording.analysis_data.significant_emotions[0]?.emotion)}
              </p>
            </div>

            {/* Chart container remains unchanged */}
            <div className="chart-container">
              <Bar
                data={{
                  labels: recording.analysis_data.significant_emotions?.map(item => item.emotion) || ['No Data'],
                  datasets: [{
                    label: 'Emotion Percentage',
                    data: recording.analysis_data.significant_emotions?.map(item => item.percentage) || [100],
                    backgroundColor: [
                      'rgba(241, 196, 15, 0.7)',  // Yellow
                      'rgba(52, 152, 219, 0.7)',  // Blue
                      'rgba(231, 76, 60, 0.7)',   // Red
                      'rgba(155, 89, 182, 0.7)',  // Purple
                      'rgba(26, 188, 156, 0.7)',  // Teal
                      'rgba(22, 160, 133, 0.7)',  // Dark teal
                      'rgba(149, 165, 166, 0.7)', // Gray
                    ]
                  }]
                }}
                options={chartOptions} // Use the simplified options
              />
            </div>
          </div>

          {recording.analysis_data.emotion_journey && (
            <div className="mt-6">
              {/* Title: White */}
              <h3 className="font-semibold mb-2 text-white">Emotional Journey</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                  {/* Sub-title: White */}
                  <h4 className="text-sm font-medium text-white">Beginning</h4>
                  <ul className="text-sm">
                    {/* Content: Light */}
                    {Object.entries(recording.analysis_data.emotion_journey.beginning || {}).map(([emotion, value]) => (
                      <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                    ))}
                  </ul>
                </div>
                <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                  {/* Sub-title: White */}
                  <h4 className="text-sm font-medium text-white">Middle</h4>
                  <ul className="text-sm">
                    {/* Content: Light */}
                    {Object.entries(recording.analysis_data.emotion_journey.middle || {}).map(([emotion, value]) => (
                      <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                    ))}
                  </ul>
                </div>
                <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                  {/* Sub-title: White */}
                  <h4 className="text-sm font-medium text-white">End</h4>
                  <ul className="text-sm">
                    {/* Content: Light */}
                    {Object.entries(recording.analysis_data.emotion_journey.end || {}).map(([emotion, value]) => (
                      <li key={emotion} className="text-neutral-lightest">{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {recording.analysis_data.interpretation && (
            <div className="mt-6">
              {/* Title: White */}
              <h3 className="font-semibold mb-2 text-white">Analysis</h3>
              <div className="p-4 bg-opacity-10 bg-white rounded">
                {/* Content: Light */}
                {Array.isArray(recording.analysis_data.interpretation) ?
                  recording.analysis_data.interpretation.map((point, index) => (
                    <p key={index} className="text-neutral-lightest mb-2">{point}</p>
                  )) :
                  <p className="text-neutral-lightest">{recording.analysis_data.interpretation}</p>
                }
              </div>
            </div>
          )}

          {recording.analysis_data.educational_recommendations && (
            <div className="mt-4">
              {/* Title: White */}
              <h3 className="font-semibold mb-2 text-white">Recommendations</h3>
              <div className="p-4 bg-opacity-10 bg-white rounded">
                {/* Content: Light */}
                {Array.isArray(recording.analysis_data.educational_recommendations) ?
                  recording.analysis_data.educational_recommendations.map((rec, index) => (
                    <p key={index} className="text-neutral-lightest mb-2">{rec}</p>
                  )) :
                  <p className="text-neutral-lightest">Based on the emotional data, maintaining the current teaching approach appears effective.</p>
                }
              </div>
            </div>
          )}
        </Card>
      )}

      <div className="mt-6">
        <button onClick={() => navigate('/history')} className="btn btn-primary">
          <i className="fas fa-arrow-left"></i> Back to History
        </button>
      </div>
    </div>
  );
};

export default RecordingDetail;