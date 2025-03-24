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

  // Prepare chart data for emotions
  const chartData = {
    labels: recording.analysis_data.significant_emotions?.map(item => item.emotion) || [],
    datasets: [
      {
        label: 'Emotion Percentage',
        data: recording.analysis_data.significant_emotions?.map(item => item.percentage) || [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(199, 199, 199, 0.6)',
        ],
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="recording-detail">
      <h2>Recording Details</h2>
      <div className="recording-meta">
        <p>Date: {moment(recording.timestamp).format('DD-MM-YYYY')}</p>
        <p>Time: {moment(recording.timestamp).format('h:mm:ss A')}</p>
      </div>

      <div className="charts-container">
        <div className="emotion-chart">
          <h3>Emotions Detected</h3>
          <Bar data={chartData} />
        </div>
      </div>

      <div className="analysis-section">
        <h3>Analysis Summary</h3>
        <div className="summary-text">
          {recording.analysis_data.summary || "No summary available."}
        </div>

        <h3>Recommendations</h3>
        <div className="recommendations">
          {recording.analysis_data.recommendations?.map((rec, index) => (
            <div key={index} className="recommendation-item">
              {rec}
            </div>
          )) || "No recommendations available."}
        </div>
      </div>

      <button onClick={() => navigate('/history')} className="back-btn">
        Back to History
      </button>
    </div>
  );
};

export default RecordingDetail;