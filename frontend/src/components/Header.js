import React from 'react';

const Header = () => {
  return (
    <header className="header">
      <div className="header-container">
        <div className="header-logo">
          <span className="logo-icon">
            <i className="fas fa-brain"></i>
          </span>
          <h1>EmotionWave</h1>
        </div>
        <p>Emotion Intelligence for Enhanced Learning</p>
      </div>
    </header>
  );
};

export default Header;