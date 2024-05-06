import React, { useState } from 'react';

const Button2 = ({ label, style, onClick }) => {
  const [isHovered, setHovered] = useState(false);

  const handleMouseEnter = () => {
    setHovered(true);
  };

  const handleMouseLeave = () => {
    setHovered(false);
  };

  return (
    <button
      style={{
        ...styles.button,
        ...style,
        ...(isHovered && styles.buttonHovered),
        
      }}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {label}
    </button>
  );
};

const styles = {
  buttonHovered: {
    backgroundColor: '#B9D9EB',
  },
  button: {
    padding: '0.5rem 1rem',
    marginTop: '10px',
    marginRight: '10px',
    backgroundColor: 'lightsteelblue',
    fontSize: '16px',
    color: '#fff',
    border: '1px solid black',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
    fontFamily: 'Raleway, sans-serif',
    width: '150px'
  },
};

export default Button2;