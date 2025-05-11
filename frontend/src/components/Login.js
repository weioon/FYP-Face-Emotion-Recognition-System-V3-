import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Login = ({ setIsAuthenticated }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    // Define apiUrl using the environment variable
    const apiUrl = process.env.REACT_APP_API_URL;

    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      // Use the apiUrl variable in the axios call
      const response = await axios.post(`${apiUrl}/api/token`, formData);
      
      // Save token to localStorage
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('username', username);
      
      // Update authentication state
      setIsAuthenticated(true);
      navigate('/dashboard');
    } catch (error) {
      setError('Invalid username or password');
      console.error('Login error:', error);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-image">
        <img src="https://i.imgur.com/RjPQDSk.jpg" alt="Emotion Analysis Technology" />
        <div className="auth-image-overlay">
          <div className="auth-image-content">
            <h2>Welcome to EmotionWave</h2>
            <p>Enhancing education through emotion intelligence</p>
          </div>
        </div>
      </div>
      <div className="auth-container">
        <div className="auth-logo">
          <i className="fas fa-brain"></i>
          <h1>EmotionWave</h1>
        </div>
        <h2>Welcome Back</h2>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <div className="input-with-icon">
              <i className="fas fa-user"></i>
              <input 
                id="username"
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                required
              />
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="input-with-icon">
              <i className="fas fa-lock"></i>
              <input 
                id="password"
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password" 
                required
              />
            </div>
          </div>
          <button type="submit" className="btn-auth">Login</button>
        </form>
        <p className="auth-switch">
          Don't have an account? <span onClick={() => navigate('/register')} className="link">Register here</span>
        </p>
      </div>
    </div>
  );
};

export default Login;