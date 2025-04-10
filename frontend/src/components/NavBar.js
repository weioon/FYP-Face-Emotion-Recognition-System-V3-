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
      <div className="nav-logo">
        <Link to="/dashboard">
          <i className="fas fa-brain nav-logo-icon"></i>
          EmotionWave
        </Link>
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
        <span className="username">
          <i className="fas fa-user-circle user-icon"></i>
          {username}
        </span>
        <button onClick={handleLogout} className="logout-btn">Logout</button>
      </div>
    </nav>
  );
};

export default NavBar;