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
      const apiPrefix = process.env.REACT_APP_API_URL === undefined ? "" : process.env.REACT_APP_API_URL;
      const requestUrl = `${apiPrefix}/api/recording_history`;
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          navigate('/login');
          return;
        }

        const response = await axios.get(requestUrl, {
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
        if (error.config && error.config.url) {
          console.error('Attempted URL:', error.config.url);
        } else {
          console.error('Attempted URL (constructed):', requestUrl);
        }
      }
    };

    fetchRecordings();
  }, [navigate]);

  const viewRecording = (recordingId) => {
    navigate(`/recording/${recordingId}`);
  };

  // Add this new function in the RecordingHistory component
  const deleteRecording = async (recordingId) => {
    const apiPrefix = process.env.REACT_APP_API_URL === undefined ? "" : process.env.REACT_APP_API_URL;
    const requestUrl = `${apiPrefix}/api/recording/${recordingId}`;
    if (window.confirm("Are you sure you want to delete this recording? This action cannot be undone.")) {
      try {
        const token = localStorage.getItem('token');
        await axios.delete(requestUrl, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        // Remove the deleted recording from state and adjust pagination
        setRecordings(prevRecordings => {
          const updatedRecordings = prevRecordings.filter(recording => recording.id !== recordingId);
          
          // Recalculate current page's items based on updated recordings
          const newIndexOfLastRecord = currentPage * recordsPerPage;
          const newIndexOfFirstRecord = newIndexOfLastRecord - recordsPerPage;
          // Ensure indices are not negative
          const effectiveIndexOfFirstRecord = Math.max(0, newIndexOfFirstRecord);
          
          const newCurrentPageRecordings = updatedRecordings.slice(effectiveIndexOfFirstRecord, newIndexOfLastRecord);

          if (newCurrentPageRecordings.length === 0 && currentPage > 1) {
            setCurrentPage(currentPage - 1);
          } else if (updatedRecordings.length === 0) { // All recordings deleted
            setCurrentPage(1);
          }
          return updatedRecordings;
        });
        
      } catch (error) {
        setError('Failed to delete recording');
        console.error('Error deleting recording:', error);
        if (error.config && error.config.url) {
          console.error('Attempted URL for delete:', error.config.url);
        } else {
          console.error('Attempted URL for delete (constructed):', requestUrl);
        }
      }
    }
  };

  // Helper function to get dominant emotion from multi-face data structure
  const getDominantEmotion = (analysisData) => {
    // If it's the new multi-face format
    if (analysisData.faces && analysisData.faces.length > 0) {
      // Get the first face's dominant emotion
      return analysisData.faces[0].dominant_emotion || 'Neutral';
    }
    
    // Fall back to old format or default
    if (analysisData.significant_emotions && analysisData.significant_emotions.length > 0) {
      return analysisData.significant_emotions[0]?.emotion || 'Neutral';
    }
    
    // If no data available
    return analysisData.dominant_emotion || 'No data';
  };

  // Pagination controls
  const paginate = (pageNumber) => setCurrentPage(pageNumber);
  const nextPage = () => setCurrentPage(prev => Math.min(prev + 1, totalPages));
  const prevPage = () => setCurrentPage(prev => Math.max(prev - 1, 1));

  // Helper function to generate visible page numbers
  const getPageNumbers = () => {
    const pageNumbers = [];
    // Always generate all page numbers from 1 to totalPages
    for (let i = 1; i <= totalPages; i++) {
      pageNumbers.push(i);
    }
    return pageNumbers;
  };

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
                          <i className="far fa-calendar-alt"></i>
                          {moment.utc(recording.timestamp).local().format('DD-MM-YYYY')}
                        </div>
                      </td>
                      <td>
                        <div className="time-cell">
                          <i className="far fa-clock"></i>
                          {moment.utc(recording.timestamp).local().format('h:mm:ss A')}
                        </div>
                      </td>
                      <td>
                        {recording.analysis_data.duration ? 
                          `${Math.round(recording.analysis_data.duration)} sec` : 
                          'N/A'}
                      </td>
                      <td>
                        {getDominantEmotion(recording.analysis_data)}
                      </td>
                      <td>
                        <div className="action-buttons">
                          <button 
                            onClick={() => viewRecording(recording.id)}
                            className="view-btn-table"
                          >
                            <i className="fas fa-eye"></i> View
                          </button>
                          <button 
                            onClick={(e) => {
                              e.stopPropagation(); // Prevent row click from triggering
                              deleteRecording(recording.id);
                            }}
                            className="delete-btn-table"
                          >
                            <i className="fas fa-trash-alt"></i> Delete
                          </button>
                        </div>
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
                  aria-label="Previous page"
                >
                  <i className="fas fa-chevron-left"></i> Prev
                </button>
                
                <div className="page-numbers">
                  {/* Make sure all page numbers are displayed */}
                  {getPageNumbers().map(number => (
                    <button
                      key={number}
                      onClick={() => paginate(number)}
                      className={`page-number ${currentPage === number ? 'active' : ''}`}
                      aria-label={`Page ${number}`}
                      aria-current={currentPage === number ? 'page' : undefined}
                    >
                      {number}
                    </button>
                  ))}
                </div>
                
                <button 
                  onClick={nextPage} 
                  disabled={currentPage === totalPages}
                  className="pagination-btn"
                  aria-label="Next page"
                >
                  Next <i className="fas fa-chevron-right"></i>
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