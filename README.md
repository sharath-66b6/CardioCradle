# Heart Disease Prediction - Production Docker Setup

A production-ready, fully containerized heart disease prediction system with separate Streamlit frontend and Flask API backend services.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Trained models and test data (see training section below)

### Run the Application

```bash
# Clone the repository
git clone <repository-url>
cd heart-disease-prediction

# Download trained models and test data
# From: https://www.dropbox.com/scl/fi/duxvgv34csb38hl0dh3iy/model_testdata.zip
# Extract to models/ and data/ directories

# Build and run with Docker Compose
docker-compose up --build

# Access the applications:
# - Streamlit UI: http://localhost:8501
# - Flask API: http://localhost:9696
```

## 📋 API Usage

### Health Check
```bash
curl http://localhost:9696/health
```

### Make Predictions
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bmi": 28.5,
    "physicalhealth": 0,
    "mentalhealth": 0,
    "sleeptime": 8,
    "smoking": "no",
    "alcoholdrinking": "no",
    "stroke": "no",
    "diffwalking": "no",
    "sex": "female",
    "agecategory": "45-49",
    "race": "white",
    "diabetic": "no",
    "physicalactivity": "yes",
    "genhealth": "good",
    "asthma": "no",
    "kidneydisease": "no",
    "skincancer": "no"
  }'
```

### Sample Response
```json
{
  "heartdisease_probability": 0.123,
  "heartdisease_prediction": false,
  "risk_level": "low",
  "recommendation": "Maintain healthy lifestyle",
  "risk_factors": ["smoking"],
  "model_used": "models/logistic_original.bin"
}
```

## 🏗️ Architecture

```
heart-disease-prediction/
├── streamlit_app/          # Streamlit frontend
│   └── app.py
├── api/                    # Flask API backend
│   └── predict.py
├── models/                 # Trained model files (.bin)
├── data/                   # Test data (df_test.pkl)
├── Dockerfile.streamlit    # Streamlit container
├── Dockerfile.api          # API container
├── docker-compose.yml      # Multi-service orchestration
└── requirements.txt        # Python dependencies
```

### Services

- **Streamlit Frontend** (Port 8501)
  - Interactive web UI for predictions
  - Model selection and comparison
  - Sample/manual data input
  - Risk visualization and metrics

- **Flask API Backend** (Port 9696)
  - RESTful prediction endpoint
  - Health check endpoint
  - Production-ready with Gunicorn
  - CORS enabled for frontend communication

## 🛠️ Development Setup

### Train Models Locally

1. **Download Dataset**
   ```bash
   # Get heart_2020_cleaned.csv from Kaggle
   # https://www.kaggle.com/datasets/luyezhang/heart-2020-cleaned
   ```

2. **Train Models**
   ```bash
   python train.py
   ```

3. **Evaluate Models** (Optional)
   ```bash
   python evaluate_models.py
   ```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py

# Run API (in another terminal)
MODEL_FILE=models/logistic_original.bin python api/predict.py
```

## 🔧 Configuration

### Environment Variables

**API Service:**
- `MODEL_FILE`: Path to model file (default: `models/logistic_original.bin`)

**Streamlit Service:**
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

### Adding New Models

1. Train and save your model using the training script
2. Update `MODEL_FILE` environment variable in `docker-compose.yml`
3. Rebuild and restart services

```yaml
environment:
  - MODEL_FILE=models/your_new_model.bin
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | Prediction Time |
|-------|----------|-----------|--------|----------|-----------------|
| lightgbm_original | 0.914 | 0.577 | 0.078 | 0.137 | 0.491s |
| logistic_original | 0.914 | 0.542 | 0.100 | 0.169 | 0.020s |
| xgboost_original | 0.913 | 0.522 | 0.096 | 0.163 | 0.122s |
| decision_tree_original | 0.911 | 0.459 | 0.080 | 0.137 | 0.040s |
| random_forest_original | 0.901 | 0.337 | 0.133 | 0.190 | 4.019s |

*Dataset: 63,959 samples, 8.74% positive class*

## 🔍 Features

### Input Features
- **Numerical**: BMI, Physical Health, Mental Health, Sleep Time
- **Categorical**: Smoking, Alcohol, Stroke, Walking Difficulty, Sex, Age Category, Race, Diabetic, Physical Activity, General Health, Asthma, Kidney Disease, Skin Cancer

### Risk Assessment
- **Low Risk**: < 30% probability
- **Medium Risk**: 30-70% probability
- **High Risk**: > 70% probability

### Model Types
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Decision Tree
- Each with/without SMOTE balancing

## 🐳 Docker Commands

```bash
# Build services
docker-compose build

# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs streamlit
docker-compose logs api

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build --force-recreate
```

## 📈 Monitoring

### Health Checks
- Streamlit: `http://localhost:8501/healthz`
- API: `http://localhost:9696/health`

### Logs
```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Service-specific logs
docker-compose logs -f streamlit
docker-compose logs -f api
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test with Docker Compose
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
- Streamlit & Flask communities
- Scikit-learn, XGBoost, LightGBM maintainers

```

```
