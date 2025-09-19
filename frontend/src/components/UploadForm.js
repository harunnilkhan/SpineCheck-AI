import React, { useState } from 'react';
import { Form, Button, Alert, Card, Spinner } from 'react-bootstrap';
import { FaUpload, FaImage } from 'react-icons/fa';
import '../styles/UploadForm.css';

const UploadForm = ({ onUpload, isLoading }) => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setError('');

    const validTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    if (selectedFile && !validTypes.includes(selectedFile.type)) {
      setError('Please select a valid image file (JPEG, PNG or BMP)');
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an X-ray image file');
      return;
    }
    onUpload(file);
  };

  return (
    <Card className="upload-form-card">
      <Card.Body>
        <Card.Title className="text-center mb-4">
          <FaImage className="me-2" />
          Upload X-ray Image
        </Card.Title>

        {error && <Alert variant="danger">{error}</Alert>}

        <Form onSubmit={handleSubmit}>
          <div className="mb-4">
            {/* label’in for=uploader input’u tetikliyor */}
            <label htmlFor="fileInput" className="upload-area">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="X-ray preview"
                  className="preview-image"
                />
              ) : (
                <div className="upload-placeholder">
                  <FaUpload size={32} />
                  <p>Click to select an X-ray image or drag and drop it here</p>
                </div>
              )}
            </label>
            <input
              type="file"
              id="fileInput"
              accept="image/jpeg,image/png,image/bmp"
              onChange={handleFileChange}
              disabled={isLoading}
              style={{ display: 'none' }}
            />
          </div>

          <div className="d-grid gap-2">
            <Button
              variant="primary"
              type="submit"
              disabled={!file || isLoading}
              className="upload-button"
            >
              {isLoading ? (
                <>
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                    className="me-2"
                  />
                  Image being analysed...
                </>
              ) : (
                <>Analyse X-ray Image</>
              )}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default UploadForm;