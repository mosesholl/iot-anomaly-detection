import pandas as pd
import requests
import time
import os

API_URL = "http://localhost:5000/predict"
DATA_PATH = os.path.join("data", "synthetic_iot_dataset_challenging.csv")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Anomaly": "is_anomaly"})

    # Take first N rows to simulate a stream
    N = 20
    subset = df.head(N)

    for idx, row in subset.iterrows():
        payload = {
            "records": [
                {
                    "Temperature": float(row["Temperature"]),
                    "Humidity": float(row["Humidity"]),
                    "Battery_Level": float(row["Battery_Level"]),
                }
            ]
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"[{idx}] Error calling API: {e}")
            continue

        print(f"[{idx}] Input: {payload['records'][0]}")
        print(f"[{idx}] Response: {result}")
        print("-" * 60)

        # Simulate a delay between sensor readings
        time.sleep(1)


if __name__ == "__main__":
    main()
