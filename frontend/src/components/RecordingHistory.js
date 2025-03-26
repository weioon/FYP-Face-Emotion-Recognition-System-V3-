import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import moment from 'moment';

const RecordingHistory = () => {
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const recordsPerPage = 10;

  // Calculate page data
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  const currentRecordings = recordings.slice(indexOfFirstRecord, indexOfLastRecord);
  const totalPages = Math.ceil(recordings.length / recordsPerPage);

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

  // Pagination controls
  const paginate = (pageNumber) => setCurrentPage(pageNumber);
  const nextPage = () => setCurrentPage(prev => Math.min(prev + 1, totalPages));
  const prevPage = () => setCurrentPage(prev => Math.max(prev - 1, 1));

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="recording-history">
      <h2>Your Recording History</h2>
      {recordings.length === 0 ? (
        <p>No recordings found. Start a new recording session!</p>
      ) : (
        <>
          <div className="recordings-list">
            {currentRecordings.map((recording) => (
              <div key={recording.id} className="recording-item" onClick={() => viewRecording(recording.id)}>
                <div className="recording-time">
                  <div className="date-container">
                    <i className="far fa-calendar-alt time-icon"></i>
                    <span className="date">{moment.utc(recording.timestamp).local().format('DD-MM-YYYY')}</span>
                  </div>
                  <div className="time-container">
                    <i className="far fa-clock time-icon"></i>
                    <span className="time">{moment.utc(recording.timestamp).local().format('h:mm:ss A')}</span>
                  </div>
                </div>
                <div className="recording-preview">
                  {recording.analysis_data.significant_emotions && recording.analysis_data.significant_emotions.length > 0 ? (
                    <div>
                      Primary emotion: {recording.analysis_data.significant_emotions[0]?.emotion || 'Neutral'}
                      {recording.analysis_data.duration && (
                        <div className="duration">
                          Session duration: {Math.round(recording.analysis_data.duration)} seconds
                        </div>
                      )}
                    </div>
                  ) : (
                    <div>
                      No significant emotions detected
                    </div>
                  )}
                </div>
                <div className="view-btn">View Details</div>
              </div>
            ))}
          </div>
          
          {recordings.length > recordsPerPage && (
            <div className="pagination">
              <button 
                onClick={prevPage} 
                disabled={currentPage === 1}
                className="pagination-btn"
              >
                Prev
              </button>
              
              <div className="page-numbers">
                {Array.from({ length: totalPages }, (_, i) => (
                  <button
                    key={i + 1}
                    onClick={() => paginate(i + 1)}
                    className={`page-number ${currentPage === i + 1 ? 'active' : ''}`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
              
              <button 
                onClick={nextPage} 
                disabled={currentPage === totalPages}
                className="pagination-btn"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default RecordingHistory;