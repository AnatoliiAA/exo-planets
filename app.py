import os

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS # Import CORS for cross-origin requests

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:5173",
    "https://cozy-rugelach-587c41.netlify.app"
])

# --- Configuration ---
MODEL_PATH = './saved_pipelines/rf_pipeline_full.joblib'

# Define the order of features
FEATURE_ORDER = ['dec_deg', 'equilibrium_temperature_k', 'insolation_flux_earth_flux',
                 'orbital_period_days', 'planet_radius_earth_radii', 'ra_deg',
                 'stellar_distance_pc', 'stellar_effective_temperature_k',
                 'stellar_logg', 'stellar_radius_solar_radii']

# --- Model Loading ---
try:
    full_pipeline = joblib.load(MODEL_PATH)
    preprocessor_pipeline = full_pipeline.named_steps['preprocessor']
    classifier_model = full_pipeline.named_steps['classifier']
    print(f"Model and preprocessor loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    full_pipeline = None
    preprocessor_pipeline = None
    classifier_model = None

FEATURE_RANGES = {
    'dec_deg': {'min': -90.0, 'max': 90.0},
    'equilibrium_temperature_k': {'min': 10.0, 'max': 10000.0},
    'insolation_flux_earth_flux': {'min': 0.0, 'max': 10000.0},
    'orbital_period_days': {'min': 0.1, 'max': 100000.0},
    'planet_radius_earth_radii': {'min': 0.1, 'max': 30.0},
    'ra_deg': {'min': 0.0, 'max': 360.0},
    'stellar_distance_pc': {'min': 0.1, 'max': 1000.0},
    'stellar_effective_temperature_k': {'min': 2000.0, 'max': 50000.0},
    'stellar_logg': {'min': 0.0, 'max': 5.0},
    'stellar_radius_solar_radii': {'min': 0.1, 'max': 1000.0}
}

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    if not full_pipeline:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    missing_features = [f for f in FEATURE_ORDER if f not in data]
    if missing_features:
        return jsonify({
            "error": "Missing features in request",
            "missing": missing_features,
            "expected": FEATURE_ORDER
        }), 400

    try:
        input_values = [float(data[feature]) for feature in FEATURE_ORDER]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data type for one or more features: {e}"}), 400

    input_df = pd.DataFrame([input_values], columns=FEATURE_ORDER)
    input_data_dict = {feature: float(data[feature]) for feature in FEATURE_ORDER}

    invalid_inputs = []
    for feature, value in input_data_dict.items():
        if feature in FEATURE_RANGES:
            min_val = FEATURE_RANGES[feature]['min']
            max_val = FEATURE_RANGES[feature]['max']
            if not (min_val <= value <= max_val):
                invalid_inputs.append(
                    f"'{feature}' with value {value} is outside expected range ({min_val} to {max_val})"
                )

    if invalid_inputs:
        return jsonify({
            "error": "Input validation failed: One or more features are outside plausible physical ranges.",
            "details": invalid_inputs
        }), 400

    try:
        processed_input = preprocessor_pipeline.transform(input_df)

        prediction_label = classifier_model.predict(processed_input)[0]
        prediction_proba = classifier_model.predict_proba(processed_input)[0]

        confidence = prediction_proba[prediction_label] * 100

        result_text = "Exoplanet Confirmed!" if prediction_label == 1 else "Not an Exoplanet."

        return jsonify({
            "prediction": result_text,
            "is_exoplanet": bool(prediction_label),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """Simple health check endpoint."""
    if full_pipeline:
        return jsonify({"status": "Model loaded and ready", "model_loaded": True}), 200
    else:
        return jsonify({"status": "Model not loaded", "model_loaded": False}), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)