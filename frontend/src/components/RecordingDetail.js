import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import moment from 'moment';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const RecordingDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [recording, setRecording] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
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

  // Check which property contains the summary data - use a more comprehensive approach
  const getSummaryData = () => {
    // Check if there's an explicit summary field
    if (recording.analysis_data.summary) {
      return recording.analysis_data.summary;
    }
    
    // Check if there's a dominant emotion field
    if (recording.analysis_data.dominant_emotion) {
      return `The dominant emotion detected was ${recording.analysis_data.dominant_emotion}`;
    }
    
    // Check if we have significant emotions data
    if (recording.analysis_data.significant_emotions && 
        recording.analysis_data.significant_emotions.length > 0) {
      
      const emotions = recording.analysis_data.significant_emotions;
      let summary = `The primary emotion detected was ${emotions[0].emotion} (${emotions[0].percentage.toFixed(1)}%)`;
      
      if (emotions.length > 1) {
        summary += `, followed by ${emotions[1].emotion} (${emotions[1].percentage.toFixed(1)}%)`;
      }
      
      // Add information about emotional stability if available
      if (recording.analysis_data.emotion_stability !== undefined) {
        const stability = recording.analysis_data.emotion_stability > 0.7 ? 'stable' : 
                          (recording.analysis_data.emotion_stability > 0.4 ? 'moderately changing' : 'highly variable');
        summary += `. The emotional pattern was ${stability} throughout the session.`;
      }
      
      return summary;
    }
    
    // Fallback if no data is available
    return "Summary analysis could not be generated due to insufficient emotion data.";
  };

  // Replace the simple summaryData variable with this function call
  const summaryData = getSummaryData();

  // Prepare chart data for emotions
  const chartData = {
    labels: recording.analysis_data.significant_emotions?.map(item => item.emotion) || ['No Data'],
    datasets: [
      {
        label: 'Emotion Percentage',
        data: recording.analysis_data.significant_emotions?.map(item => item.percentage) || [100],
        backgroundColor: [
          'rgba(22, 255, 0, 0.8)',  // Green - matches your accent
          'rgba(0, 179, 255, 0.8)', // Blue
          'rgba(255, 107, 107, 0.8)', // Coral red
          'rgba(156, 114, 255, 0.8)', // Purple
          'rgba(0, 207, 187, 0.8)', // Teal
          'rgba(155, 164, 180, 0.8)', // Gray
          'rgba(255, 149, 81, 0.8)', // Orange
        ],
        borderColor: [
          'rgba(22, 255, 0, 1)',
          'rgba(0, 179, 255, 1)',
          'rgba(255, 107, 107, 1)',
          'rgba(156, 114, 255, 1)',
          'rgba(0, 207, 187, 1)',
          'rgba(155, 164, 180, 1)',
          'rgba(255, 149, 81, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  // Add chart options for better visibility
  const chartOptions = {
    plugins: {
      legend: {
        labels: {
          color: '#FFFFFF',
          font: {
            size: 14
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(10, 16, 34, 0.8)',
        titleColor: '#FFFFFF',
        bodyColor: '#FFFFFF',
        bodyFont: {
          size: 14
        },
        padding: 10,
        boxPadding: 5,
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1
      }
    },
    scales: {
      y: {
        ticks: {
          color: '#FFFFFF',
          font: {
            size: 12
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      x: {
        ticks: {
          color: '#FFFFFF',
          font: {
            size: 12
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    }
  };

  return (
    <div className="recording-detail">
      {/* Recording Header Section */}
      <div className="recording-header">
        <h2>Recording Details</h2>
        <div className="recording-meta">
          <p>Date: {moment.utc(recording.timestamp).local().format('DD-MM-YYYY')}</p>
          <p>Time: {moment.utc(recording.timestamp).local().format('h:mm:ss A')}</p>
          <p>Duration: {recording.analysis_data.duration ? `${Math.round(recording.analysis_data.duration)} seconds` : 'N/A'}</p>
        </div>
      </div>

      {/* Chart Section */}
      <div className="charts-container">
        <h3>
          <i className="fas fa-chart-pie"></i>
          Emotions Detected
        </h3>
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* Analysis Summary Section */}
      <div className="summary-section">
        <h3>
          <i className="fas fa-chart-line"></i>
          Analysis Summary
        </h3>
        <div className="summary-text">
          {summaryData}
        </div>
      </div>
      
      {/* Interpretation Section */}
      <div className="interpretation-section">
        <h3>
          <i className="fas fa-brain"></i>
          Interpretation
        </h3>
        <div className="interpretation">
          {recording.analysis_data.interpretation || "The emotional state was relatively consistent throughout the session."}
        </div>
      </div>

      {/* Emotional Journey Section */}
      <div className="journey-section">
        <h3>
          <i className="fas fa-chart-area"></i>
          Emotional Journey
        </h3>
        <div className="journey-stages">
          <div className="stage">
            <h4>Beginning</h4>
            <ul>
              {Object.entries(recording.analysis_data.emotion_journey?.beginning || {}).map(([emotion, value]) => (
                <li key={emotion}>{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
              ))}
            </ul>
          </div>
          <div className="stage">
            <h4>Middle</h4>
            <ul>
              {Object.entries(recording.analysis_data.emotion_journey?.middle || {}).map(([emotion, value]) => (
                <li key={emotion}>{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
              ))}
            </ul>
          </div>
          <div className="stage">
            <h4>End</h4>
            <ul>
              {Object.entries(recording.analysis_data.emotion_journey?.end || {}).map(([emotion, value]) => (
                <li key={emotion}>{emotion}: {typeof value === 'number' ? value.toFixed(1) : value}%</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Recommendations Section */}
      <div className="recommendations-section">
        <h3>
          <i className="fas fa-lightbulb"></i>
          Educational Recommendations
        </h3>
        <div className="recommendations">
          {(recording.analysis_data.educational_recommendations && recording.analysis_data.educational_recommendations.length > 0) ? 
            recording.analysis_data.educational_recommendations.map((rec, index) => (
              <div key={index} className="recommendation-item">
                {rec}
              </div>
            )) : 
            (recording.analysis_data.recommendations && recording.analysis_data.recommendations.length > 0) ?
            recording.analysis_data.recommendations.map((rec, index) => (
              <div key={index} className="recommendation-item">
                {rec}
              </div>
            )) : 
            <div className="recommendation-item">
              Based on the emotional data, maintaining the current teaching approach appears effective.
            </div>
          }
        </div>
      </div>

      <button onClick={() => navigate('/history')} className="back-btn">
        <i className="fas fa-arrow-left"></i> Back to History
      </button>
    </div>
  );
};

export default RecordingDetail;