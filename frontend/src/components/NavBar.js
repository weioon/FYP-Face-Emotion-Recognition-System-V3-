import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const NavBar = ({ setIsAuthenticated }) => {
  const navigate = useNavigate();
  const username = localStorage.getItem('username');

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setIsAuthenticated(false);
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="nav-logo">
        <Link to="/dashboard">Face Emotion Recognition</Link>
      </div>
      <div className="nav-links">
        <Link to="/dashboard" className={location.pathname === "/dashboard" ? "active" : ""}>
          <i className="nav-icon fas fa-chart-bar"></i>
          Dashboard
        </Link>
        <Link to="/history" className={location.pathname === "/history" ? "active" : ""}>
          <i className="nav-icon fas fa-history"></i>
          History
        </Link>
      </div>
      <div className="nav-user">
        <span className="username">{username}</span>
        <button onClick={handleLogout} className="logout-btn">Logout</button>
      </div>
    </nav>
  );
};

export default NavBar;