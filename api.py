from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("model", "iot_anomaly_model.joblib")
ANOMALY_THRESHOLD = 0.5  # default threshold for class 1 probabilities

app = Flask(__name__)

model = None


def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "threshold": ANOMALY_THRESHOLD,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "records" not in data:
        return jsonify({"error": "Request JSON must contain 'records' list"}), 400

    records = data["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    df = pd.DataFrame(records)

    # Model now expects Device_ID + numeric features
    expected_cols = ["Temperature", "Humidity", "Battery_Level", "Device_ID"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required feature(s): {missing}"}), 400

    # Ensure we only pass the columns the model was trained on
    X = df[expected_cols].copy()

    # Make sure Device_ID is string
    X["Device_ID"] = X["Device_ID"].astype(str)

    try:
        probas = model.predict_proba(X)[:, 1]  # probability of class 1 (anomaly)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

    flags = probas >= ANOMALY_THRESHOLD

    results = []
    for p, flag in zip(probas, flags):
        results.append(
            {
                "is_anomaly": bool(flag),
                "anomaly_probability": float(p),
            }
        )

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
