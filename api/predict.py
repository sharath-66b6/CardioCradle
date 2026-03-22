import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model from environment variable
MODEL_FILE = os.getenv('MODEL_FILE', 'models/logistic_original.bin')

try:
    with open(MODEL_FILE, 'rb') as f:
        dv, scaler, model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}")
    dv, scaler, model = None, None, None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if dv is None or scaler is None or model is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'model_file': MODEL_FILE
        }), 503

    return jsonify({
        'status': 'healthy',
        'message': 'API is running',
        'model_file': MODEL_FILE
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for heart disease risk"""
    try:
        # Check if model is loaded
        if dv is None or scaler is None or model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': f'Failed to load model from {MODEL_FILE}'
            }), 500

        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide patient data in JSON format'
            }), 400

        # Validate required features
        required_features = [
            'bmi', 'physicalhealth', 'mentalhealth', 'sleeptime',
            'smoking', 'alcoholdrinking', 'stroke', 'diffwalking',
            'sex', 'agecategory', 'race', 'diabetic',
            'physicalactivity', 'genhealth', 'asthma',
            'kidneydisease', 'skincancer'
        ]

        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features,
                'message': f'Please provide all required features: {required_features}'
            }), 400

        # Convert to DataFrame for processing
        patient_df = pd.DataFrame([data])

        # Preprocess numerical features
        numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
        X_num = scaler.transform(patient_df[numerical_features])

        # Preprocess categorical features
        categorical_dict = {k: v for k, v in data.items() if k not in numerical_features}
        X_cat = dv.transform([categorical_dict])

        # Combine features
        X = np.hstack([X_num, X_cat])

        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0, 1])
            prediction = bool(probability >= 0.5)
        else:
            prediction = bool(model.predict(X)[0])
            probability = float(prediction)

        # Determine risk level
        if probability >= 0.7:
            risk_level = 'high'
            recommendation = 'Immediate medical consultation recommended'
        elif probability >= 0.3:
            risk_level = 'medium'
            recommendation = 'Regular health check-ups advised'
        else:
            risk_level = 'low'
            recommendation = 'Maintain healthy lifestyle'

        # Identify risk factors
        risk_factors = []
        if data.get('smoking') == 'yes':
            risk_factors.append('smoking')
        if data.get('alcoholdrinking') == 'yes':
            risk_factors.append('alcohol_consumption')
        if data.get('stroke') == 'yes':
            risk_factors.append('previous_stroke')
        if data.get('diffwalking') == 'yes':
            risk_factors.append('difficulty_walking')
        if data.get('diabetic') in ['yes', 'yes_(during_pregnancy)']:
            risk_factors.append('diabetes')
        if data.get('genhealth') in ['fair', 'poor']:
            risk_factors.append('poor_general_health')

        # Return response
        response = {
            'heartdisease_probability': probability,
            'heartdisease_prediction': prediction,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'model_used': MODEL_FILE
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=False)
