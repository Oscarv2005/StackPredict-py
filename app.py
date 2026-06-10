import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).parent

# Vercel environments have a read-only root filesystem. 
# If running on Vercel, we redirect dynamic file writing to the writable /tmp directory.
if os.environ.get('VERCEL'):
    logger.info("Vercel environment detected. Routing model artifacts to /tmp")
    MODEL_PATH = Path("/tmp") / "model.pkl"
    FEATURES_PATH = Path("/tmp") / "features.pkl"
else:
    MODEL_PATH = BASE_DIR / "model.pkl"
    FEATURES_PATH = BASE_DIR / "features.pkl"

# Read-only assets remain anchored to the base project directory
CSV_PATH = BASE_DIR / "online_shoppers_intention.csv"
EDA_PATH = BASE_DIR / "eda_special_day.png"

model = None
feature_names = None
model_info = {}

def train_model():
    """Trains a clean linear regression model on continuous tracking features."""
    try:
        logger.info(f"Loading data from {CSV_PATH}")
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Dataset file missing at path: {CSV_PATH}")
        
        df = pd.read_csv(CSV_PATH)
        
        if 'PageValues' not in df.columns:
            raise ValueError("Target column 'PageValues' missing from dataset context.")
        
        # Isolate explicitly mapped interface attributes to ensure 1:1 structural matching
        base_features = [
            "Administrative", "Administrative_Duration", 
            "Informational", "Informational_Duration", 
            "ProductRelated", "ProductRelated_Duration", 
            "BounceRates", "ExitRates"
        ]
        
        X = df[base_features].copy()
        y = df['PageValues'].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model_obj = LinearRegression()
        model_obj.fit(X_train, y_train)
        
        y_pred = model_obj.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Save to resolved paths (handles Vercel's /tmp switch gracefully)
        joblib.dump(model_obj, MODEL_PATH)
        joblib.dump(list(X.columns), FEATURES_PATH)
        
        info = {
            "status": "trained",
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "features_count": len(X.columns),
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        }
        return model_obj, list(X.columns), info
    except Exception as e:
        logger.error(f"Error during model training sequence: {e}")
        raise

def load_or_train_model():
    global model, feature_names, model_info
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        logger.info("Loading pre-trained analytical models...")
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        model_info = {"status": "loaded", "features_count": len(feature_names)}
    else:
        logger.info("Pre-trained analytical models not found. Executing training suite...")
        model, feature_names, model_info = train_model()

try:
    load_or_train_model()
except Exception as e:
    logger.critical(f"Critical failure during initialization vector: {e}")

@app.route('/', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "error", "message": "Model framework structural collapse."}), 500
    return jsonify({"status": "success", "model_info": model_info}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_names is None:
        return jsonify({"status": "error", "message": "Prediction engine offline."}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing input data structure payload."}), 400
        
        row = {}
        for feat in feature_names:
            try:
                row[feat] = float(data.get(feat, 0))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": f"Attribute parsing failure on field '{feat}'"}), 400

        # Enforce statistical range constraints
        if row.get("BounceRates", 0) < 0 or row.get("BounceRates", 0) > 1:
            return jsonify({"status": "error", "message": "Bounce rate input validation error: Value must fall between 0 and 1"}), 400
        if row.get("ExitRates", 0) < 0 or row.get("ExitRates", 0) > 1:
            return jsonify({"status": "error", "message": "Exit rate input validation error: Value must fall between 0 and 1"}), 400
        
        X_input = pd.DataFrame([row], columns=feature_names)
        prediction = max(0, float(model.predict(X_input)[0]))
        
        return jsonify({
            "status": "success",
            "predicted_page_value": prediction,
            "model_info": f"Linear Regression (R² = {model_info.get('r2_score', 0.0):.4f})"
        }), 200
    except Exception as e:
        logger.error(f"Execution error on prediction array: {e}")
        return jsonify({"status": "error", "message": "Downstream model computation execution crash."}), 500

# Required for Vercel serverless execution parsing
app_obj = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)