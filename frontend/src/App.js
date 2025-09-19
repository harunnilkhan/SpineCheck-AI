import React, { useState } from 'react';
import { Container, Row, Col, Alert } from 'react-bootstrap';
import axios from 'axios';
import Header from './components/Header';
import UploadForm from './components/UploadForm';
import ResultView from './components/ResultView';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Handle file upload and analysis
  const handleUpload = async (file) => {
    setLoading(true);
    setError('');

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing image:', err);

      if (err.response) {
        setError(`Hata: ${err.response.data.detail || 'Image could not be analysed'}`);
      } else if (err.request) {
        setError('Hata: The server is not responding. Please try again later.');
      } else {
        setError(`Hata: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Reset state to analyze another image
  const handleReset = () => {
    setResult(null);
    setError('');
  };

  return (
    <div className="app">
      <Header />
      <Container className="main-container">
        {error && (
          <Alert variant="danger" className="mb-4">
            {error}
          </Alert>
        )}

        {!result ? (
          <Row className="justify-content-center">
            <Col xs={12} md={8} lg={6}>
              <UploadForm onUpload={handleUpload} isLoading={loading} />
            </Col>
          </Row>
        ) : (
          <Row className="justify-content-center">
            <Col xs={12} lg={10}>
              <ResultView result={result} onReset={handleReset} />
            </Col>
          </Row>
        )}
      </Container>
      <Footer />
    </div>
  );
}

export default App;