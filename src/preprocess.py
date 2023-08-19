import numpy as np
import pandas as pd
from config import RESULT_DATA_PATH, DATA_ML_PATH
import logging
import zipfile
import os
from typing import List
import tqdm
import tempfile

logging.basicConfig(level=logging.INFO)

def reduce_mem_usage(data):
    """ Функция для оптимизации использования памяти DataFrame (inplace). """
    
    # Расчет начального использования памяти -
    start_memory = data.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {start_memory:.2f} MB")
    
    # Создание словарей с диапазонами для каждого типа чисел
    int_type_dict = {
        (np.iinfo(np.int8).min,  np.iinfo(np.int8).max):  np.int8,
        (np.iinfo(np.int16).min, np.iinfo(np.int16).max): np.int16,
        (np.iinfo(np.int32).min, np.iinfo(np.int32).max): np.int32,
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max): np.int64,
    }
    
    float_type_dict = {
        (np.finfo(np.float16).min, np.finfo(np.float16).max): np.float16,
        (np.finfo(np.float32).min, np.finfo(np.float32).max): np.float32,
        (np.finfo(np.float64).min, np.finfo(np.float64).max): np.float64,
    }
    
    # Обрабатываем каждый столбец в DataFrame
    for column in data.columns:
        col_type = data[column].dtype

        if np.issubdtype(col_type, np.integer):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in int_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)
        elif np.issubdtype(col_type, np.floating):
            c_min = data[column].min()
            c_max = data[column].max()
            dtype = next((v for k, v in float_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                data[column] = data[column].astype(dtype)
    
    # Расчет конечного использования памяти
    end_memory = data.memory_usage().sum() / 1024**2
    print(f"Final memory usage: {end_memory:.2f} MB")
    print(f"Reduced by {(start_memory - end_memory) / start_memory * 100:.1f}%")

def process_parquet_dataset_from_zip(zip_path: str, columns: List[str] = None, verbose: bool = False) -> pd.DataFrame:
    # Создание временной директории
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Извлечение файлов формата parquet из zip-архива
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Получение словаря с путями к файлам parquet
        dataset_paths = {
            int(os.path.splitext(filename)[0].split("_")[-1]): os.path.join(tmp_dir, 'train_data', filename)
            for filename in os.listdir(os.path.join(tmp_dir, 'train_data')) if filename.endswith('.pq')
        }

        # Сортировка путей к файлам по номеру партиции
        chunks = [dataset_paths[num] for num in sorted(dataset_paths.keys())]

        dataframes = []

        # Если verbose=True, вывод информации о путях к файлам
        if verbose:
            logging.info("Processing chunks:\n" + "\n".join(chunks))

        for chunk_path in tqdm.tqdm(chunks, desc="Processing dataset with pandas"):
            df = pd.read_parquet(chunk_path, columns=columns)
            reduce_mem_usage(df)
            dataframes.append(df)
            del df

        result = pd.concat(dataframes, ignore_index=True)
        result.to_parquet(RESULT_DATA_PATH, index=False)
        return result


def check_and_drop_columns(data, threshold=80):
    """Удаляет неинформативные столбцы."""
    drop_columns = []

    for column in data.columns:
        percentage = (data[column].value_counts().max() / len(data[column])) * 100
        if percentage >= threshold:
            drop_columns.append(column)
            logging.info(f"Column '{column}' deleted ({percentage:6.2f}%)")

    return data.drop(columns=drop_columns)


def process_and_save_chunks(df, chunk_ratio=0.01):
    """ 
    Разбивает DataFrame на части, преобразует и сохраняет каждую часть, 
    затем загружает, оптимизирует и объединяет все части.
    """
    df = check_and_drop_columns(df)
    # Создаем директорию 'data_ohe', если она отсутствует
    output_directory = "data_ohe"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Разделение DataFrame на части
    chunk_size = int(df.shape[0] * chunk_ratio)
    chunks = [x for x in range(0, df.shape[0], chunk_size)]
    processed_files = []

    for start in chunks:
        end = start + chunk_size
        chunk = df.iloc[start:end]
        chunk_ids = chunk['id']

        # Преобразование и агрегирование данных, исключая признак 'id'
        chunk_without_id = chunk.drop(columns=['id'])
        chunk_transformed = pd.get_dummies(chunk_without_id, drop_first=True)
        chunk_aggregated = chunk_transformed.groupby(chunk_ids).sum().reset_index()

        # Сохранение обработанного фрагмента на диск
        filename = os.path.join(output_directory, f"processed_chunk_{start}_{end}.parquet")
        chunk_aggregated.to_parquet(filename, index=False)
        processed_files.append(filename)

    # Загрузка, оптимизация и объединение обработанных частей
    final_df = []

    for filename in processed_files:
        df_chunk = pd.read_parquet(filename)
        reduce_mem_usage(df_chunk)
        final_df.append(df_chunk)
        # Опционально: удаление временных файлов после их загрузки
        os.remove(filename)

    final_df = pd.concat(final_df, axis=0).reset_index(drop=True)
    final_df.to_csv(DATA_ML_PATH, index=False)
    return final_df

