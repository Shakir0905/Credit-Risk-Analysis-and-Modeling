import requests
import pandas as pd
from config import DATA_ML_PATH

url = "http://127.0.0.1:8000/predict"
df = pd.read_csv(DATA_ML_PATH, nrows=5)

# Конвертация в список словарей с исключением 'id' из 'data'
items = [{"id": row['id'], "data": {k: v for k, v in row.items() if k != "id"}} for row in df.to_dict(orient='records')]

response = requests.post(url, json=items)

print('Status:', response.status_code)
for result in response.json():
    print(f"Client ID: {result['client_id']}, Prediction: {result['prediction']}")