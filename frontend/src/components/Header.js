import React from 'react';
import { Navbar, Container } from 'react-bootstrap';
import { FaSpinner } from 'react-icons/fa';
import '../styles/Header.css';

const Header = () => {
  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="header-navbar">
      <Container>
        <Navbar.Brand href="/">
          <FaSpinner className="spin-icon" />
          <span className="brand-text">SpineCheck-AI</span>
        </Navbar.Brand>
        <Navbar.Text className="ms-auto text-light">
          Detection of scoliosis from X-ray images
        </Navbar.Text>
      </Container>
    </Navbar>
  );
};

export default Header