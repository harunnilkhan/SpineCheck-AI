import React from 'react';
import { Container } from 'react-bootstrap';
import '../styles/Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <Container className="text-center">
        <p className="mb-0">
          SpineCheck-AI &copy; {new Date().getFullYear()} |
            A research tool for detecting scoliosis from X-ray images
        </p>
      </Container>
    </footer>
  );
};

export default Footer;