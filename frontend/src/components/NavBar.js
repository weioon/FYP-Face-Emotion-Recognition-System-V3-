import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';

const NavBar = ({ setIsAuthenticated }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const username = localStorage.getItem('username');

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setIsAuthenticated(false);
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="nav-brand">
          <Link to="/dashboard">
            <i className="fas fa-brain"></i>
          </Link>
        </div>

        <div className="system-name">
          <h1>EmotionWave</h1>
        </div>
        
        <div className="nav-controls">
          <div className="nav-buttons">
            <Link to="/dashboard" className={`nav-btn ${location.pathname === "/dashboard" ? "active" : ""}`}>
              <i className="fas fa-chart-bar"></i>
              <span>Dashboard</span>
            </Link>
            
            <Link to="/history" className={`nav-btn ${location.pathname === "/history" ? "active" : ""}`}>
              <i className="fas fa-history"></i>
              <span>History</span>
            </Link>
          </div>

          <div className="nav-user">
            <span className="username">
              <i className="fas fa-user-circle"></i>
              {username}
            </span>
            <button onClick={handleLogout} className="logout-btn">
              <i className="fas fa-sign-out-alt"></i>
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;