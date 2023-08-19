from pathlib import Path

# Configuration for the data processing and model training
BASE_DIR = Path(__file__).parent.parent  # Определение корневой директории вашего проекта
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'train_data.zip'
TARGET_PATH = BASE_DIR / 'data' / 'raw' / 'train_target.csv'
RESULT_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'result_data.parquet'
DATA_ML_PATH = BASE_DIR / 'data' / 'processed' / 'data_for_ml.csv'
MODEL_PATH = BASE_DIR / 'models' / 'model.pkl'
