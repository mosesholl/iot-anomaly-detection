# Anomaly Detection in IoT Sensor Data

This project implements a simple **anomaly detection system** for IoT sensor data in a factory-like environment.  
It was built as part of a university course project (oral project report, “From Model to Production”).

The system consists of:

- A **trained machine learning model** that detects anomalies in IoT sensor readings  
- A **RESTful API** (Flask) that exposes the model via HTTP  
- A **simulated sensor client** that streams data to the API  
- A **Jupyter/Colab notebook** documenting data exploration and model training  
- A **small test script** to easily test the `/predict` endpoint  

---

## 1. Project overview

### Problem

Imagine a factory where IoT devices are attached to machines. The devices measure:

- **Temperature**
- **Humidity**
- **Battery level** (used as a proxy for machine/device health)

The goal is to detect **anomalous sensor readings** in (near) real time so that potential failures can be found early.

### Approach

1. Use a **synthetic IoT dataset** with labeled anomalies.  
2. Train a **binary classifier** (`is_anomaly` ∈ {0, 1}) in a scikit-learn pipeline.  
3. Save the trained model as a `.joblib` file.  
4. Load the model in a **Flask API** and expose a `/predict` endpoint.  
5. Simulate a sensor stream with a small Python client that sends HTTP requests to the API.  

---

## 2. Tech stack

- **Language:** Python 3  
- **Core libraries:**
  - `pandas`, `numpy` – data handling  
  - `scikit-learn` – modeling (RandomForest, Pipeline, etc.)  
  - `joblib` – saving/loading the model  
  - `flask` – REST API  
  - `requests` – HTTP client for the simulated sensor stream and tests  
- **Environment:**
  - Developed using Google Colab (for data & model)  
  - Local development with VS Code + virtual environment  

---

## 3. Repository structure

```text
.
├── README.md                      # Project description (this file)
├── requirements.txt               # Python dependencies
├── api.py                         # Flask REST API (loads and serves the model)
├── stream_client.py               # Simulated sensor client (sends data to the API)
├── test_predict.py                # Small script to test /predict
├── data/
│   └── synthetic_iot_dataset_challenging.csv   # Synthetic IoT dataset
├── model/
│   └── iot_anomaly_model.joblib   # Trained scikit-learn Pipeline (model + scaler)
└── notebooks/
    └── iot_anomaly_detection_colab.ipynb       # Notebook: EDA + training + model export


```


## 4. Dataset

**File:** `data/synthetic_iot_dataset_challenging.csv`  

The dataset contains **3,000** synthetic IoT records with the following columns:

- `Device_ID` – ID of the simulated device (e.g. DHT11_A, DHT11_B, …)  
- `Temperature` – normalized temperature reading (float)  
- `Humidity` – normalized humidity reading (float)  
- `Battery_Level` – normalized battery level (float)  
- `Anomaly` – integer label (`0 = normal`, `1 = anomaly`)  

For modeling, the notebook:

- Renames `Anomaly` → `is_anomaly`  
- Uses the three sensor columns as features:
  - `Temperature`, `Humidity`, `Battery_Level`  
- Uses `is_anomaly` as the target label  

---

## 5. Model

Model training is documented in:

`notebooks/iot_anomaly_detection_colab.ipynb`

Main steps:

1. Load and inspect `synthetic_iot_dataset_challenging.csv`.  
2. Perform basic EDA: class balance, summary statistics, distributions.  
3. Rename `Anomaly` → `is_anomaly`.  
4. Create feature matrix `X` and target vector `y`:
   - `X = [Temperature, Humidity, Battery_Level]`  
   - `y = is_anomaly`  
5. Train/test split (e.g. 80% / 20%, stratified).  
6. Define a scikit-learn **Pipeline**:

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])

7. Train the model and evaluate it with:
   - Classification report (precision, recall, F1)  
   - Confusion matrix  

8. Fit on the full dataset and save the trained pipeline as:

    model/iot_anomaly_model.joblib

The API loads this pipeline and uses it to predict anomalies.

---

## 6. Setup & installation

### 6.1 Prerequisites

- Python 3.9+ (Python 3 recommended)  
- `pip`  
- A terminal (Command Prompt, PowerShell, or zsh/bash)  

### 6.2 Get the code

You can either:

**Option A – Clone (if you have git):**

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

**Option B – Download ZIP:**

1. Click the green **Code** button on GitHub → “Download ZIP”.  
2. Unzip it.  
3. Open a terminal in the unzipped folder.  

### 6.3 Create and activate a virtual environment (recommended)

**macOS / Linux:**

    python3 -m venv .venv
    source .venv/bin/activate

**Windows (PowerShell):**

    python -m venv .venv
    .\.venv\Scripts\Activate

You should see `(.venv)` at the start of your terminal prompt.

### 6.4 Install dependencies

With the virtual environment active:

    pip install --upgrade pip
    pip install -r requirements.txt

This installs Flask, scikit-learn, joblib, pandas, numpy, requests, etc.

---

## 7. Running the API

### 7.1 Start the Flask server

From the project root (where `api.py` is located), with your virtual environment active:

**macOS / Linux:**

    python3 api.py

**Windows:**

    python api.py

You should see something like:

    * Serving Flask app 'api'
    * Debug mode: on
    * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)

The API is now available at **http://localhost:5000**.

### 7.2 Health check

Open a browser and visit:

    http://localhost:5000/health

You should get a JSON response similar to:

    {
      "status": "ok",
      "model_loaded": true
    }

This means:

- The Flask app is running.  
- The model file `model/iot_anomaly_model.joblib` was loaded successfully.  

---

## 8. Prediction endpoint

### 8.1 Endpoint: `POST /predict`

- Accepts one or multiple sensor records in JSON format.  
- Returns an anomaly flag and an anomaly probability for each record.  

**Example request JSON:**

    {
      "records": [
        {
          "Device_ID": "DHT11_A",
          "Temperature": 0.1,
          "Humidity": -0.2,
          "Battery_Level": 0.5
        }
      ]
    }

**Example response JSON:**

    {
      "results": [
        {
          "is_anomaly": false,
          "anomaly_probability": 0.12
        }
      ]
    }

> The exact probability will vary depending on the trained model.

### 8.2 Test with curl (terminal)

With the API running:

    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "records": [
          {
            "Temperature": 0.1,
            "Humidity": -0.2,
            "Battery_Level": 0.5
          }
        ]
      }'

### 8.3 Test with Python (`test_predict.py`)

You can run the test script:

    python3 test_predict.py

The script sends one test record to `/predict` and prints the response.

**Example `test_predict.py`:**

    import requests

    url = "http://localhost:5000/predict"

    payload = {
        "records": [
            {
                "Temperature": 0.1,
                "Humidity": -0.2,
                "Battery_Level": 0.5,
            }
        ]
    }

    response = requests.post(url, json=payload, timeout=10)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())

---

## 9. Simulated sensor stream

The file `stream_client.py` simulates a **data stream** of sensor readings to the API.

### How it works

1. Loads the dataset from `data/synthetic_iot_dataset_challenging.csv`.  
2. Renames `Anomaly` to `is_anomaly` (if needed).  
3. Iterates over the first N rows (e.g. 20).  
4. For each row:
   - Builds a JSON payload with `Temperature`, `Humidity`, `Battery_Level`.  
   - Sends it to `http://localhost:5000/predict` via `requests.post`.  
   - Prints the API response.  
   - Waits 1 second to simulate a real-time flow.  

### Run the stream client

With the API still running in one terminal, open another terminal (same folder, same virtual environment):

    python3 stream_client.py

You should see output similar to:

    [0] Input: {'Temperature': -0.55, 'Humidity': -0.15, 'Battery_Level': 0.37}
    [0] Response: {'results': [{'is_anomaly': False, 'anomaly_probability': 0.07}]}
    ------------------------------------------------------------
    [1] Input: {...}
    [1] Response: {...}
    ...

This demonstrates an end-to-end “mini streaming” system.

---
## 10. Architecture overview

Conceptually, the system looks like this:

```text
+------------------+      +--------------------------+      +----------------------------+
|  IoT Sensors     | ---> |  Ingestion / Client      | ---> |  Anomaly Detection Service |
| (Temperature,    |      | (stream_client.py)       |      |  (Flask API + ML model)    |
|  Humidity,       |      |                          |      |                            |
|  Battery_Level)  |      |  - Reads CSV             |      |  - Loads joblib model      |
|                  |      |  - Sends JSON via HTTP   |      |  - /predict endpoint       |
+------------------+      +--------------------------+      +----------------------------+
                                                                    |
                                                                    v
                                                         +----------------------+
                                                         | Alerts / Dashboards |
                                                         | (conceptual)        |
                                                         +----------------------+
```

- **Sensors** (conceptual) generate readings.  
- **Client** simulates ingestion by sending HTTP POST requests.  
- **API** scores each record as anomaly / normal.  
- In a real deployment, responses could feed alerting or dashboard systems.  

---

## 11. Development & retraining workflow

1. **Data + model development**  
   - Use `notebooks/iot_anomaly_detection_colab.ipynb`:
     - Explore data  
     - Train and evaluate models  
     - Save the best model to `model/iot_anomaly_model.joblib`  

2. **Serving / deployment**  
   - Ensure the model file is in `model/`.  
   - Start the API (`api.py`).  
   - Use `stream_client.py` or other clients to call the API.  

3. **Updating the model**  
   - If new data becomes available:
     - Retrain in the notebook (or a new script).  
     - Overwrite `model/iot_anomaly_model.joblib` with the updated pipeline.  
     - Restart the API to load the new model.  

This separation follows a simple **train → export → serve** pattern.

---

## 12. Known limitations & future work

**Limitations:**

- Synthetic dataset, not real industrial data.  
- Basic model (RandomForest) with simple hyperparameters.  
- No authentication, authorization, or rate limiting on the API.  
- No persistent logging or monitoring stack (e.g. Prometheus, Grafana).  

**Possible improvements:**

- Integrate more sensors (e.g. vibration, noise).  
- Perform model selection & hyperparameter tuning.  
- Add configurable thresholds for anomaly alerts.  
- Add logging of requests and predictions to a database.  
- Add authentication tokens or API keys.  
- Package the API as a Docker container and deploy to a cloud environment.





