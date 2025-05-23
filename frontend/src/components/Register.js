import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      setIsLoading(false);
      return;
    }

    try {
      // Define apiUrl using the environment variable
      const apiUrl = process.env.REACT_APP_API_URL;

      const payload = {
        username: username,
        email: email,
        password: password,
      };

      // Use the apiUrl variable in the axios call
      const response = await axios.post(`${apiUrl}/api/register`, payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      console.log("Registration successful:", response.data);
      // Redirect to login page after successful registration
      navigate('/login');
    } catch (error) {
      console.error('Registration error:', error);
      if (error.response) {
        // The server responded with a status code outside the 2xx range
        setError(error.response.data?.detail || 'Registration failed');
      } else if (error.request) {
        // The request was made but no response was received
        setError('No response from server. Please try again later.');
      } else {
        // Something happened in setting up the request
        setError('An error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-image">
        <img src="https://i.imgur.com/oQvzMdK.jpg" alt="Educational Emotion Analysis" />
        <div className="auth-image-overlay">
          <div className="auth-image-content">
            <h2>Join EmotionWave</h2>
            <p>Transform your teaching with emotion analytics</p>
          </div>
        </div>
      </div>
      <div className="auth-container">
        <div className="auth-logo">
          <i className="fas fa-brain"></i>
          <h1>EmotionWave</h1>
        </div>
        <h2>Create Account</h2>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleRegister}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <div className="input-with-icon">
              <i className="fas fa-user"></i>
              <input 
                id="username"
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Choose a username"
                required
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <div className="input-with-icon">
              <i className="fas fa-envelope"></i>
              <input 
                id="email"
                type="email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
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
                placeholder="Create a password"
                required
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm Password</label>
            <div className="input-with-icon">
              <i className="fas fa-lock"></i>
              <input 
                id="confirmPassword"
                type="password" 
                value={confirmPassword} 
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your password"
                required
              />
            </div>
          </div>
          
          <button type="submit" disabled={isLoading} className="btn-auth">
            {isLoading ? 'Registering...' : 'Register'}
          </button>
        </form>
        
        <p className="auth-switch">
          Already have an account? <span onClick={() => navigate('/login')} className="link">Login here</span>
        </p>
      </div>
    </div>
  );
};

export default Register;