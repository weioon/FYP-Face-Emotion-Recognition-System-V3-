import React, { useEffect, useRef } from 'react'; // Import useEffect and useRef
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import Card from './Card';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const EmotionDashboard = ({ analysisResults, isRecording }) => {
  const dashboardRef = useRef(null); // Create a ref for the main container

  // Effect to scroll down when analysis results are ready
  useEffect(() => {
    // Check if analysisResults exist, we are NOT currently recording, and the ref is attached
    if (analysisResults && !isRecording && dashboardRef.current) {
      // Scroll the dashboard container into view smoothly
      dashboardRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    // Dependency array: run this effect when analysisResults or isRecording changes
  }, [analysisResults, isRecording]);

  if (isRecording) {
    return (
      // Assign the ref to the main container div for the recording message
      <div ref={dashboardRef}>
        <Card title="Recording Session" variant="primary">
          <div className="recording-message">
            <h3 className="text-center text-xl font-semibold mb-2">Recording in Progress</h3>
            <p className="text-center text-gray-600">Capturing emotional data for analysis...</p>
            <div className="recording-indicator"></div>
          </div>
        </Card>
      </div>
    );
  }

  if (!analysisResults) {
    return (
      // Assign the ref to the main container div for the no-data message
      <div ref={dashboardRef}>
        <Card title="Emotion Analysis" variant="default">
          <div className="no-data-message text-center py-8">
            <h3 className="text-xl font-semibold mb-2">No Analysis Data</h3>
            <p className="text-gray-600">Start a recording session to see emotion analysis results.</p>
          </div>
        </Card>
      </div>
    );
  }

  // Handle multi-face results
  if (analysisResults.faces && analysisResults.faces.length > 0) {
    return (
      // Assign the ref to the main container div for the results display
      <div ref={dashboardRef} className="emotion-dashboard space-y-6">
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
                  'rgba(155, 89, 182, 0.7)',  // Fear - Purple
                  'rgba(26, 188, 156, 0.7)',  // Surprise - Teal
                  'rgba(22, 160, 133, 0.7)',  // Disgust - Dark Teal (adjust color)
                  'rgba(149, 165, 166, 0.7)', // Neutral - Gray
                ],
                borderColor: [ // Optional: Add border colors
                  'rgba(241, 196, 15, 1)',
                  'rgba(52, 152, 219, 1)',
                  'rgba(231, 76, 60, 1)',
                  'rgba(155, 89, 182, 1)',
                  'rgba(26, 188, 156, 1)',
                  'rgba(22, 160, 133, 1)',
                  'rgba(149, 165, 166, 1)',
                ],
                borderWidth: 1
              }
            ]
          };

          // --- Use the EXACT SAME chart options as RecordingDetail.js ---
          const chartOptions = {
            responsive: true,
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  color: 'var(--neutral-lightest)',
                }
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                  color: 'var(--neutral-lightest)',
                },
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              },
              x: {
                ticks: {
                  color: 'var(--neutral-lightest)',
                },
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              }
            }
          };
          // --- End of chart options ---

          return (
            <Card key={faceData.face_id || index} title={`Person ${index + 1} (ID: ${faceData.face_id})`} variant="primary">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left side: Overview and Chart */}
                <div>
                  <div className="mb-4">
                    <h3 className="font-semibold mb-2">Session Overview</h3>
                    <p className="text-neutral-lightest">
                      <span className="font-medium">Duration:</span> {faceData.duration?.toFixed(2)} seconds
                    </p>
                    <p className="text-neutral-lightest">
                      <span className="font-medium">Dominant Emotion:</span> {faceData.dominant_emotion}
                    </p>
                  </div>
                  <div className="chart-container">
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                </div>

                {/* Right side: Journey, Interpretation, Recommendations */}
                <div>
                  {faceData.emotion_journey && (
                    <div className="mb-6">
                      <h3 className="font-semibold mb-2">Emotional Journey</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          <h4 className="text-sm font-medium text-neutral-lightest">Beginning</h4>
                          <ul className="text-sm">
                            {Object.entries(faceData.emotion_journey.beginning).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {value.toFixed(1)}%</li>
                            ))}
                          </ul>
                        </div>
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          <h4 className="text-sm font-medium text-neutral-lightest">Middle</h4>
                          <ul className="text-sm">
                            {Object.entries(faceData.emotion_journey.middle).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {value.toFixed(1)}%</li>
                            ))}
                          </ul>
                        </div>
                        <div className="journey-phase p-4 bg-opacity-10 bg-white rounded">
                          <h4 className="text-sm font-medium text-neutral-lightest">End</h4>
                          <ul className="text-sm">
                            {Object.entries(faceData.emotion_journey.end).map(([emotion, value]) => (
                              <li key={emotion} className="text-neutral-lightest">{emotion}: {value.toFixed(1)}%</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}

                  {faceData.interpretation && (
                    <div className="mb-4">
                      <h3 className="font-semibold mb-2">Analysis</h3>
                      <div className="p-4 bg-opacity-10 bg-white rounded">
                        {Array.isArray(faceData.interpretation) ?
                          faceData.interpretation.map((point, i) => (
                            <p key={i} className="text-neutral-lightest mb-2">{point}</p>
                          )) :
                          <p className="text-neutral-lightest">{faceData.interpretation}</p>
                        }
                      </div>
                    </div>
                  )}

                  {faceData.educational_recommendations && (
                    <div>
                      <h3 className="font-semibold mb-2">Recommendations</h3>
                      <div className="p-4 bg-opacity-10 bg-white rounded">
                        <ul className="list-disc pl-5">
                          {faceData.educational_recommendations.map((rec, i) => (
                            <li key={i} className="text-neutral-lightest">{rec}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    );
  }

  // Fallback for potential old single-face format (if needed)
  // Assign the ref to the main container div here too
  return (
    <div ref={dashboardRef} className="emotion-dashboard space-y-6">
      {/* Render single-face analysis if necessary */}
      <Card title="Emotion Analysis" variant="primary">
        {/* ... content for single face analysis ... */}
        <p>Displaying legacy single-face analysis data...</p>
      </Card>
    </div>
  );
};

export default EmotionDashboard;