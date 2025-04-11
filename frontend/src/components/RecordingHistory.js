import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import moment from 'moment';
import Card from './Card';

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
    <>
      <Card title="Recording History" variant="primary">
        {recordings.length === 0 ? (
          <div className="no-data-message">
            <p>No recordings found. Start a new recording session!</p>
          </div>
        ) : (
          <>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Duration</th>
                    <th>Major Emotion</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {currentRecordings.map((recording) => (
                    <tr key={recording.id}>
                      <td>
                        <div className="date-cell">
                          <i className="far fa-calendar-alt mr-2"></i>
                          {moment.utc(recording.timestamp).local().format('DD-MM-YYYY')}
                        </div>
                      </td>
                      <td>
                        <div className="time-cell">
                          <i className="far fa-clock mr-2"></i>
                          {moment.utc(recording.timestamp).local().format('h:mm:ss A')}
                        </div>
                      </td>
                      <td>
                        {recording.analysis_data.duration ? 
                          `${Math.round(recording.analysis_data.duration)} sec` : 
                          'N/A'}
                      </td>
                      <td>
                        {recording.analysis_data.significant_emotions && 
                         recording.analysis_data.significant_emotions.length > 0 ? 
                          recording.analysis_data.significant_emotions[0]?.emotion || 'Neutral' : 
                          'No data'}
                      </td>
                      <td>
                        <button 
                          onClick={() => viewRecording(recording.id)}
                          className="view-btn-table"
                        >
                          <i className="fas fa-eye mr-1"></i> View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
      </Card>
    </>
  );
};

export default RecordingHistory;