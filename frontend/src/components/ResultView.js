import React, { useState } from 'react';
import { Card, Row, Col, Badge, Tabs, Tab, Button } from 'react-bootstrap';
import { FaAngleRight, FaDownload, FaUndo } from 'react-icons/fa';
import '../styles/ResultView.css';

const ResultView = ({ result, onReset }) => {
  const [activeTab, setActiveTab] = useState('visualization');

  if (!result) return null;

  // Severity renk sınıfı
  const getSeverityColor = (classification) => {
    switch (classification) {
      case 'Normal':
        return 'success';
      case 'Mild Scoliosis':
        return 'info';
      case 'Moderate Scoliosis':
        return 'warning';
      case 'Severe Scoliosis':
      case 'Very Severe Scoliosis':
        return 'danger';
      default:
        return 'secondary';
    }
  };

  // Görüntü indirme fonksiyonu
  const downloadImage = (base64Data, filename) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="result-container">
      <Card className="result-card mb-4">
        <Card.Body>
          <Card.Title className="text-center mb-4">Analysis Results</Card.Title>

          <Row className="text-center mb-4">
            <Col xs={12} md={6} className="mb-3 mb-md-0">
              <div className="result-stat-container">
                <h6 className="stat-label">Cobb Angle</h6>
                <div className="stat-value">
                  {result.max_angle.toFixed(1)}°
                </div>
              </div>
            </Col>
            <Col xs={12} md={6}>
              <div className="result-stat-container">
                <h6 className="stat-label">Classification</h6>
                <div>
                  <Badge
                    bg={getSeverityColor(result.classification)}
                    className="classification-badge"
                  >
                    {result.classification}
                  </Badge>
                </div>
              </div>
            </Col>
          </Row>

          {result.cobb_angles.length > 0 && (
            <div className="mb-4">
              <h6>Detected Curves:</h6>
              <ul className="angles-list">
                {result.cobb_angles.map((angle, index) => (
                  <li key={index} className="angle-item">
                    <FaAngleRight className="me-2" />
                    <span>
                      Curve {index + 1}: {angle.angle.toFixed(1)}°
                      (Spine #{angle.vertebra1} – #{angle.vertebra2})
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <Tabs
            activeKey={activeTab}
            onSelect={(k) => setActiveTab(k)}
            className="mb-3"
          >
            <Tab eventKey="visualization" title="Analysis Visualisation">
              <div className="result-image-container text-center">
                <img
                  src={`data:image/png;base64,${result.visualization_base64}`}
                  alt="Analysis Visualisation"
                  className="result-image"
                />
                <Button
                  variant="outline-primary"
                  className="mt-3 download-btn"
                  onClick={() =>
                    downloadImage(
                      result.visualization_base64,
                      'spinecheck_analysis.png'
                    )
                  }
                >
                  <FaDownload className="me-2" />
                  Download Visualisation
                </Button>
              </div>
            </Tab>
            <Tab eventKey="segmentation" title="Segmentation Mask">
              <div className="result-image-container text-center">
                <img
                  src={`data:image/png;base64,${result.mask_base64}`}
                  alt="Segmentation Mask"
                  className="result-image"
                />
                <Button
                  variant="outline-primary"
                  className="mt-3 download-btn"
                  onClick={() =>
                    downloadImage(
                      result.mask_base64,
                      'spinecheck_segmentation.png'
                    )
                  }
                >
                  <FaDownload className="me-2" />
                  Download Segmentation
                </Button>
              </div>
            </Tab>
          </Tabs>

          <div className="text-center mt-4">
            <Button
              variant="secondary"
              onClick={onReset}
              className="reset-btn"
            >
              <FaUndo className="me-2" />
              Perform New Image Analysis
            </Button>
          </div>
        </Card.Body>
      </Card>

      <Card className="info-card">
        <Card.Body>
          <Card.Title className="mb-3">Scoliosis Classification</Card.Title>
          <ul className="classification-list">
          <li>
            <Badge bg="success" className="me-2">Normal</Badge> Less than 10 degrees
          </li>
          <li>
            <Badge bg="info" className="me-2">Mild</Badge> 10–24 degrees
          </li>
          <li>
            <Badge bg="warning" className="me-2">Moderate</Badge> 25–39 degrees
          </li>
          <li>
            <Badge bg="danger" className="me-2">Severe</Badge> 40–49 degrees
          </li>
          <li>
            <Badge bg="danger" className="me-2">Very Severe</Badge> 50 degrees or more
          </li>
        </ul>
          <Card.Text className="disclaimer-text mt-3">
            <strong>Warning:</strong> This application is for educational and research purposes only.
            Always consult a healthcare professional for medical diagnosis and treatment.
          </Card.Text>
        </Card.Body>
        </Card>
        </div>
        );
        };


        export default ResultView;