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

response = requests.post(url, json=payload, timeout=5)
print("Status code:", response.status_code)
print("Response JSON:", response.json())
