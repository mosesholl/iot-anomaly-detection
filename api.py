from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Configuration
MODEL_PATH = os.path.join("model", "iot_anomaly_model.joblib")
FEATURES = ["Temperature", "Humidity", "Battery_Level"]

# Create Flask app
app = Flask(__name__)

# Load the trained model (Pipeline with scaler + classifier)
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict anomalies for one or multiple sensor records.

    Expected JSON:

    {
      "records": [
        {
          "Temperature": 0.1,
          "Humidity": -0.2,
          "Battery_Level": 0.5
        },
        ...
      ]
    }

    Returns:

    {
      "results": [
        {
          "is_anomaly": true,
          "anomaly_probability": 0.83
        },
        ...
      ]
    }
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if "records" not in data:
        return jsonify({"error": "JSON must contain 'records' key"}), 400

    records = data["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    # Validate and extract features
    X = []
    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            return jsonify({"error": f"Record at index {idx} is not an object"}), 400

        try:
            X.append([float(r[feat]) for feat in FEATURES])
        except KeyError as ke:
            return jsonify({"error": f"Missing feature {ke} in record {idx}"}), 400
        except ValueError:
            return jsonify({"error": f"Non-numeric value in record {idx}"}), 400

    X = np.array(X)

    # Model predictions
    preds = model.predict(X)

    # Some classifiers have predict_proba, some don't
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # probability of class 1 (anomaly)
    else:
        # Fallback: no probabilities available
        probs = np.zeros_like(preds, dtype=float)

    results = []
    for p, prob in zip(preds, probs):
        results.append(
            {
                "is_anomaly": bool(p),
                "anomaly_probability": float(prob),
            }
        )

    return jsonify({"results": results})


if __name__ == "__main__":
    # Run development server
    app.run(host="0.0.0.0", port=5000, debug=True)
