import React from 'react';
import './style.css';

const Card = ({ children, title, className = '', variant = 'default' }) => {
  const cardVariants = {
    default: 'bg-white',
    primary: 'border-l-4 border-primary-color',
    success: 'border-l-4 border-success-color',
    warning: 'border-l-4 border-accent-color'
  };

  return (
    <div className={`card ${cardVariants[variant]} ${className}`}>
      {title && (
        <div className="card-header">
          <h2 className="card-title">{title}</h2>
        </div>
      )}
      <div className="card-body">
        {children}
      </div>
    </div>
  );
};

export default Card;