iptsr# Refactor Heart Disease Prediction to Production-Ready Docker Setup

## Directory Structure
- [x] Create `streamlit_app/` directory
- [x] Create `api/` directory

## Streamlit App
- [x] Create `streamlit_app/app.py` - Enhanced Streamlit app with:
  - Relative paths for container compatibility
  - Better error handling
  - Improved model selection UI (name/type/SMOTE)
  - Sample/manual input toggle
  - Progress bar during prediction
  - Risk messages and metrics display

## Flask API
- [x] Create `api/predict.py` - Production Flask API with:
  - Load one model via environment variable
  - Use scaler for preprocessing
  - /predict POST endpoint with JSON input
  - /health GET endpoint
  - Proper error handling and validation

## Dependencies
- [x] Update `requirements.txt`:
  - Add flask
  - Add gunicorn
  - Remove ipykernel

## Docker Configuration
- [x] Create `Dockerfile.streamlit`:
  - python:3.11-slim base
  - Install requirements
  - Copy streamlit_app/
  - CMD streamlit run
- [x] Create `Dockerfile.api`:
  - python:3.11-slim base
  - Install requirements
  - Copy api/
  - CMD gunicorn
- [x] Create `docker-compose.yml`:
  - Two services (streamlit, api)
  - Shared volumes for models/ and data/
  - Ports 8501/9696
  - Environment variables for model file
  - Health checks

## Configuration Files
- [x] Update `.dockerignore`:
  - Add data/
  - Add models/
- [x] Update `README.md`:
  - New docker-compose instructions
  - Curl examples for API
  - Guide for adding models

## Testing and Validation
- [x] Test docker-compose build
- [x] Test docker-compose run
- [x] Verify API endpoints with curl
- [x] Ensure relative paths work in containers
- [x] Validate model loading and predictions
