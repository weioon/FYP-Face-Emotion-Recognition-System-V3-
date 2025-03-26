// filepath: frontend/src/components/Button.js
import React from 'react';

const Button = ({ children, variant = 'primary', onClick, disabled, type, className = '' }) => {
  return (
    <button 
      className={`btn btn-${variant} ${className}`}
      onClick={onClick}
      disabled={disabled}
      type={type || 'button'}
    >
      {children}
    </button>
  );
};

export default Button;