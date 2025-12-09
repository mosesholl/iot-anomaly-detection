import requests

API_URL = "http://localhost:5000/predict"

def main():
    payload = {
        "records": [
            {
                "Device_ID": "DHT11_A",     # example device ID
                "Temperature": 0.1,
                "Humidity": -0.2,
                "Battery_Level": 0.5,
            }
        ]
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        print("Status code:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except ValueError:
            print("Non-JSON response:", resp.text)
    except requests.RequestException as e:
        print("Error calling API:", e)


if __name__ == "__main__":
    main()
