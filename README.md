# SpineCheck-AI

A web-based AI application that detects spine from 2D X-ray images, calculates Cobb angle, and classifies scoliosis severity.

## Features

- 🔍 Vertebra detection from X-ray images
- 📏 Automatic Cobb angle calculation
- 🏥 Scoliosis severity classification
- 🎯 Manual annotation tool for creating training data
- 🧠 UNet segmentation model for vertebra detection
- 🖥️ User-friendly web interface

## Project Structure

```
SpineCheck-AI/
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── dataset/          # Dataset management
├── models/           # Trained models
├── tools/            # Manual annotation tool
└── training/         # Model training scripts
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Node.js 14+
- VS Code

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harunnilkhan/SpineCheck-AI.git
cd SpineCheck-AI
```

2. Set up the Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
cd ..
```

## Data Preparation

### Using the Annotation Tool

1. Place your raw X-ray images in the `dataset/raw/` directory
2. Run the annotation tool:
```bash
python tools/annotation_tool.py
```
3. Follow the tool's interface to annotate vertebrae on each X-ray image
4. The tool will save annotations in `dataset/annotations/` and binary masks in `dataset/masks/`

## Training the Model

After annotating enough X-ray images:

```bash
python training/train.py
```

This will train the UNet model and save the trained model as `models/unet_model.pth`.

## Running the Application

### Start the Backend

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### Start the Frontend

```bash
cd frontend
npm start
```

The web interface will be available at `http://localhost:3000`.

## Usage

1. Upload an X-ray image through the web interface
2. The application will:
   - Detect vertebrae using the trained model
   - Calculate the Cobb angle
   - Classify scoliosis severity
   - Display results with visualization

## Scoliosis Classification Criteria

- **Normal**: Less than 10 degrees
- **Mild Scoliosis**: 10-24 degrees
- **Moderate Scoliosis**: 25-39 degrees
- **Severe Scoliosis**: 40-49 degrees
- **Very Severe Scoliosis**: 50 degrees or more

## Disclaimer

This application is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
