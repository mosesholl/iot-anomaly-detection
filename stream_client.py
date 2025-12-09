import time
import requests
import pandas as pd

API_URL = "http://localhost:5000/predict"
CSV_PATH = "data/synthetic_iot_dataset_challenging.csv"  # adjust if needed

def main():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Could not find CSV at {CSV_PATH}")
        return

    print(f"Loaded dataset with {len(df)} rows")

    # Rename Anomaly for consistency (not used for sending)
    if "Anomaly" in df.columns:
        df = df.rename(columns={"Anomaly": "is_anomaly"})

    N = 20  # number of rows to stream
    for i, (_, row) in enumerate(df.head(N).iterrows()):
        payload = {
            "records": [
                {
                    "Device_ID": str(row["Device_ID"]),
                    "Temperature": float(row["Temperature"]),
                    "Humidity": float(row["Humidity"]),
                    "Battery_Level": float(row["Battery_Level"]),
                }
            ]
        }

        print(f"\n[{i}] Sending payload:", payload)

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            print(f"[{i}] Status code:", resp.status_code)
            try:
                print(f"[{i}] Response JSON:", resp.json())
            except ValueError:
                print(f"[{i}] Non-JSON response:", resp.text)
        except requests.RequestException as e:
            print(f"[{i}] Error calling API: {e}")

        print("-" * 60)
        time.sleep(1)


if __name__ == "__main__":
    main()

