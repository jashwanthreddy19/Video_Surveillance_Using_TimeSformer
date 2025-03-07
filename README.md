# Video Surveillance Using TimeSformer

## Overview
Video Surveillance Using TimeSformer is a project designed to detect abnormal events (such as fighting, assault, theft, etc.) in real-time from multiple CCTV feeds. The system utilizes a fine-tuned Timesformer model for video classification, processes video streams into fixed-length clips, and sends alerts when an anomaly is detected.

## Features
- **Real-Time Video Ingestion:** Accepts multiple video streams using protocols like RTSP.
- **Video Preprocessing:** Extracts frames, resizes/crops them to 224×224, and generates fixed-length clips.
- **Abnormal Event Detection:** Uses a fine-tuned Timesformer model to classify events.
- **Alert System:** Sends notifications via email/SMS/push when an abnormal event is detected.
- **Web Dashboard:** Provides a user-friendly interface for monitoring video feeds, viewing event logs, and system status.

## Project Structure
Video Surveillance Using TimeSformer/ ├── configs/ # Configuration files (model paths, hyperparameters, etc.) ├── data/ # Folder for datasets and sample videos ├── src/ # Source code │ ├── video_processing.py # Module for video ingestion and preprocessing │ ├── model_inference.py # Module for loading and running the Timesformer model │ ├── app.py # Web application (Flask or FastAPI) for serving video feeds and alerts ├── tests/ # Unit and integration tests ├── README.md # Project overview and documentation (this file) └── requirements.txt # List of project dependencies

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jashwanthreddy19/Video_Surveillance_Using_TimeSformer.git
   cd Video_Surveillance_Using_TimeSformer

Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Configuration:
Update the configuration files in the configs/ directory as needed.
Ensure dataset paths and model checkpoint paths are correctly specified.
Usage

Video Processing: 
Process videos to extract frames and create fixed-length clips:
python src/video_processing.py

Model Inference: Run the model inference module on the preprocessed clips:
python src/model_inference.py

Web Application: Start the web server to access the dashboard and view alerts:
python src/app.py

Next Steps

Environment Setup:
Verify that your local environment is working correctly by running a small OpenCV script to process sample video files.

Video Ingestion & Preprocessing Module:
Develop the module that handles reading video streams, extracting frames, and generating fixed-length clips.

Model Fine-Tuning & Integration:
Fine-tune the Timesformer model on your dataset and integrate the model into the inference module.

Web Application Development:
Build the backend API and a basic frontend dashboard for monitoring video feeds and alerts.

Testing & Deployment:
Write tests, perform integration testing, and prepare for eventual cloud deployment.

Roadmap
Phase 1: Local environment setup and video processing pipeline.
Phase 2: Fine-tuning the Timesformer model and integrating it for inference.
Phase 3: Developing and integrating the web application for real-time alerts.
Phase 4: Comprehensive testing, optimization, and scaling for cloud deployment.



Contact
For questions or contributions, please contact [jashwanthreddykolla@gmail.com] or open an issue on GitHub.