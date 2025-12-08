# Anomaly Detection in IoT Sensor Data (Task 1)

This project implements a simple anomaly detection system for IoT sensor data
in a factory-like setting. It is part of the **Oral Project Report (Task 1)**.

The system:

- Uses a synthetic IoT dataset with three sensor features:
  - `Temperature`
  - `Humidity`
  - `Battery_Level`
- Trains a binary classifier to detect anomalies (`is_anomaly` ∈ {0,1})
- Exposes the trained model via a **RESTful API** using Flask
- Includes a small **simulated sensor client** that streams data to the API

---

## Repository structure

```text
iot-anomaly-detection/
├── README.md
├── requirements.txt
├── api.py                  # Flask REST API
├── stream_client.py        # Simulated sensor client
├── model/
│   └── iot_anomaly_model.joblib      # Saved scikit-learn Pipeline
├── data/
│   └── synthetic_iot_dataset_challenging.csv
└── notebooks/
    └── iot_anomaly_detection_colab.ipynb
