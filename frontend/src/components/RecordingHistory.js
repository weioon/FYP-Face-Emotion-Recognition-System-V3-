import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import moment from 'moment';

const RecordingHistory = () => {
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchRecordings = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          navigate('/login');
          return;
        }

        const response = await axios.get('http://localhost:8000/recording_history', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        setRecordings(response.data);
        setLoading(false);
      } catch (error) {
        setError('Failed to fetch recording history');
        setLoading(false);
        console.error('Error fetching recordings:', error);
      }
    };

    fetchRecordings();
  }, [navigate]);

  const viewRecording = (recordingId) => {
    navigate(`/recording/${recordingId}`);
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="recording-history">
      <h2>Your Recording History</h2>
      {recordings.length === 0 ? (
        <p>No recordings found. Start a new recording session!</p>
      ) : (
        <div className="recordings-list">
          {recordings.map((recording) => (
            <div key={recording.id} className="recording-item" onClick={() => viewRecording(recording.id)}>
              <div className="recording-time">
                <span className="date">{moment(recording.timestamp).format('DD-MM-YYYY')}</span>
                <span className="time">{moment(recording.timestamp).format('h:mm A')}</span>
              </div>
              <div className="recording-preview">
                {/* Simple preview of emotions */}
                {recording.analysis_data.significant_emotions && (
                  <div>
                    Primary emotion: {recording.analysis_data.significant_emotions[0]?.emotion || 'N/A'}
                  </div>
                )}
              </div>
              <div className="view-btn">View Details</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RecordingHistory;