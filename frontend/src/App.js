import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

// Import components
import Login from './components/Login';
import Register from './components/Register';
import EmotionDetector from './components/EmotionDetector';
import EmotionDashboard from './components/EmotionDashboard';
import RecordingHistory from './components/RecordingHistory';
import RecordingDetail from './components/RecordingDetail';
import NavBar from './components/NavBar';
import './App.css';

// Add this after your imports in src/App.js
axios.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Auth protected route component
const ProtectedRoute = ({ children, isAuthenticated }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  return children;
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  
  // Check if user is already authenticated on app load
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // Setup axios default headers for authentication
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setIsAuthenticated(true);
    }
  }, []);

  return (
    <Router>
      <div className="app">
        {isAuthenticated && <NavBar setIsAuthenticated={setIsAuthenticated} />}
        
        <div className="content">
          <Routes>
            <Route path="/login" element={
              isAuthenticated ? 
                <Navigate to="/dashboard" /> : 
                <Login setIsAuthenticated={setIsAuthenticated} />
            } />
            
            <Route path="/register" element={
              isAuthenticated ? 
                <Navigate to="/dashboard" /> : 
                <Register />
            } />
            
            <Route path="/dashboard" element={
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <div className="dashboard-container">
                  <EmotionDetector 
                    setAnalysisResults={setAnalysisResults} 
                    isRecording={isRecording} 
                    setIsRecording={setIsRecording}
                  />
                  <EmotionDashboard 
                    analysisResults={analysisResults}
                    isRecording={isRecording} 
                  />
                </div>
              </ProtectedRoute>
            } />
            
            <Route path="/history" element={
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <RecordingHistory />
              </ProtectedRoute>
            } />
            
            <Route path="/recording/:id" element={
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <RecordingDetail />
              </ProtectedRoute>
            } />
            
            <Route path="/" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;