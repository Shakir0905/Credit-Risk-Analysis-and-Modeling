import pandas as pd
from config import DATA_ML_PATH, MODEL_PATH
import joblib

# Загрузка модели 
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_ML_PATH, nrows=2) 

# Конвертация в список словарей, где каждый словарь представляет собой строку
selected_rows = df.to_dict(orient='records')
json_data = {"data": selected_rows}
data = pd.DataFrame(json_data['data'])

# Сохранение client_id и удаление его из данных перед предсказанием
client_ids = data['id'].tolist()
data = data.drop(columns=['id'])

print(json_data)

# Получаем предсказание от модели
print('Getting predictions ...')
predictions = model.predict(data)

# Вывод результатов
for client_id, prediction in zip(client_ids, predictions):
    print(f'Client_id: {client_id}, prediction: {prediction}')

