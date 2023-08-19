# Импорт стандартных библиотек Python
import glob
import os
import warnings
import joblib
import pickle
import gc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

# Импорт библиотек для работы с данными
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from scipy.sparse import load_npz
from scipy.stats import boxcox
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, get_scorer, log_loss, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     train_test_split, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from zipfile import ZipFile

# Импорт функции display из IPython.display
from IPython.display import display

# Управление предупреждениями
warnings.filterwarnings('ignore')

import pandas as pd
import joblib
import json
from tqdm import tqdm

import os
import pandas as pd

def load_data(path: str):
    """
    Загружает данные из файла в зависимости от расширения файла.

    Args:
        path: путь к файлу.

    Returns:
        Загруженные данные (возможно, в формате DataFrame).
    """
    file_extension = os.path.splitext(path)[1][1:]  # Получение расширения файла
    loaders = {'csv': pd.read_csv, 'joblib': joblib.load, 'pkl': pd.read_pickle, 'json': pd.read_json, 'txt': lambda f: f.read()}
    if file_extension not in loaders.keys():
        raise ValueError(f"Unsupported file extension: {file_extension}")

    data = loaders[file_extension](path)

    try:
        data = pd.DataFrame(data)
    except ValueError:
        pass  # Если не можем конвертировать данные в DataFrame, оставляем их как есть

    return data


def process_data(data, method=False, constant_value=None, visualize=True, remove_duplicates=False):
    """
    Функция для обработки данных: проверки и устранения пропущенных значений, удаления дубликатов и визуализации пропущенных данных.

    Параметры:
        data (pandas.DataFrame): Исходные данные.
        method (str): Метод для заполнения пропущенных значений. Опции: 'mean', 'median', 'mode', 'constant', 'interpolation', 'dropna'. По умолчанию: False.
        constant_value (любой тип): Значение, которым будут заполнены пропущенные данные, если method='constant'.
        visualize (bool): Если True, визуализирует пропущенные данные с использованием missingno. По умолчанию: True.
        remove_duplicates (bool): Если True, удаляет дубликаты из данных. По умолчанию: False.

    Возвращает:
        data (pandas.DataFrame): Обработанные данные, если были внесены изменения. Иначе, возвращает исходные данные.
    """

    # Создаем копию данных для сохранения оригинала
    data = data.copy()

    is_changed = False

    # Заполняем или удаляем пропущенные значения
    if method is False:
        missing_values = data.isnull().sum()
        percent_missing = missing_values / data.shape[0] * 100
        print(f"The percent of missing values in data:\n\n{percent_missing}\n")
    else:
        columns = data.columns[data.isnull().any()].tolist()  # Выбираем столбцы с пропущенными значениями

        mode = data[columns].mode()
        mode_value = mode.iloc[0] if not mode.empty else data[columns].mean(numeric_only=True)  # Используем среднее, если мода пуста

        fill_values = {
            'mean': data[columns].mean(numeric_only=True),
            'median': data[columns].median(numeric_only=True),
            'mode': mode_value,
            'constant': pd.Series(constant_value, index=columns, dtype=float),
            'interpolation': data[columns].select_dtypes(include='number').interpolate(method='linear'),
            'dropna': ''  # Это значение будет использовано для удаления пропущенных значений на следующих этапах
        }.get(method)

        if method == 'constant' and constant_value is None:
            raise ValueError("Please provide 'constant_value' for the 'constant' method")

        if fill_values is None:
            raise ValueError(f"Unknown method: {method}")

        if method == 'dropna':
            data = data.dropna()
            is_changed = True
        elif method in ['mean', 'median', 'mode', 'constant', 'interpolation']:
            data = data.fillna(fill_values)
            is_changed = True

        # Проверяем пропущенные значения
        missing_values = data.isnull().sum()
        percent_missing = missing_values / data.shape[0] * 100
        print(f"The percent of missing values in data: \n\n{percent_missing}\n")

        # Удаляем дубликаты, если remove_duplicates равно True
        if remove_duplicates:
            print(f"The number of duplicates before removed: {data.duplicated().sum()}\n")
            data = data.drop_duplicates()
            is_changed = True

    print(f"The number of duplicates in data: {data.duplicated().sum()}\n")

    # Визуализируем пропущенные данные
    if visualize:
        msno.matrix(data)
        plt.title('Missing Data Pattern Matrix', fontsize=26)
        plt.show()

    # Возвращаем обработанные данные, если были внесены изменения
    if is_changed:
        return data


def plot_outliers(data, columns=None, visualize=True, skip_no_outliers=True, sample_size=500, handle_outliers=None):
    """
    Визуализирует и выводит процент выбросов для каждой числовой переменной в DataFrame.

    Параметры:
    - data: DataFrame, исходные данные
    - columns: список, содержащий названия столбцов, для которых нужно проверить выбросы (по умолчанию None)
    - visualize: bool, указывает на необходимость визуализации данных (по умолчанию True)
    - skip_no_outliers: bool, указывает на пропуск вывода для переменных без выбросов (по умолчанию True)
    - sample_size: int, количество выбираемых данных для отображения (по умолчанию 500)
    - handle_outliers: string, указывает на необходимость замены или удаления выбросов (может быть 'replace', 'remove' или None)
    """
    columns = columns or data.select_dtypes(include=['int', 'float']).columns.tolist()
    k = 2.5

    def replace_outliers(data, column, lower_bound, upper_bound):
        data.loc[data[column] < lower_bound, column] = lower_bound
        data.loc[data[column] > upper_bound, column] = upper_bound
        return data

    def remove_outliers(data, column, lower_bound, upper_bound):
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    handle_method = {'replace': replace_outliers, 'remove': remove_outliers}.get(handle_outliers)

    def calculate_outliers_percent(column):
        nonlocal data
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        if handle_method:
            data = handle_method(data, column, lower_bound, upper_bound)

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_percent = (len(outliers) / len(data)) * 100
        return outliers_percent

    outliers_percents = {column: calculate_outliers_percent(column) for column in columns}

    if visualize:
        visualize_outliers(data, outliers_percents, sample_size)

    for column, outliers_percent in outliers_percents.items():
        if skip_no_outliers and outliers_percent == 0:
            continue
        print(f"{column} - Выбросы: {outliers_percent:.2f}%")

    return data if handle_outliers in ['remove', 'replace'] else None

def visualize_outliers(data, outliers_percents, sample_size):
    """
    Визуализация выбросов в данных.

    Параметры:
    - data: DataFrame, исходные данные
    - outliers_percents: dict, содержащий процент выбросов для каждого столбца
    - sample_size: int, количество выбираемых данных для отображения
    """
    # Отфильтровываем столбцы для визуализации
    columns_to_plot = [column for column, percent in outliers_percents.items() if percent > 0]

    if columns_to_plot: # Если есть столбцы для визуализации
        # Создание фигуры и осей
        fig, ax = plt.subplots(figsize=(12, 8))

        # Построение boxplot с прозрачным цветом и разделительными линиями
        sns.boxplot(data=data[columns_to_plot], showfliers=False, ax=ax,
                    boxprops=dict(facecolor='orange', edgecolor='black', alpha=0.5),
                    whiskerprops=dict(color='black', linestyle='-'))

        # If the dataset size is less than the specified sample_size, adjust the sample_size
        adjusted_sample_size = min(sample_size, data.shape[0])

        # Выбор случайной подвыборки данных для отображения
        sampled_data = data.sample(n=adjusted_sample_size)

        # Построение stripplot для отображения выбранной подвыборки данных
        sns.stripplot(data=sampled_data[columns_to_plot], color='blue', alpha=0.3, jitter=0.2, size=4, ax=ax)

        ax.set_title('Проверка выбросов\n')
        ax.set_xlabel('Переменные')
        ax.set_ylabel('Значения')

        # Вычисляем шаг расположения текста
        step = 1 / (len(columns_to_plot) + 1)

        # Выводим надписи по каждому столбцу в отдельности
        for i, column in enumerate(columns_to_plot):
            fig.text((i + 1) * step, 0.90, f'Выбросы в {column}: {outliers_percents[column]:.2f}%', 
                    ha='center', va='center', fontsize=10, color='red')

        # Добавление горизонтальных линий
        for line in range(1, 10):
            ax.axhline(line, color='gray', linestyle='--', linewidth=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Нет столбцов с выбросами для визуализации.")



def plot_distributions_and_transform(data, columns=None, bins=80, max_categories=20, transform_method=None):
    data = data.copy()
    transform_methods = {
        'boxcox': lambda x: boxcox(x + 1)[0],
        'log': np.log1p,
        'sqrt': np.sqrt,
        'cbrt': np.cbrt
    }

    if transform_method and transform_method not in transform_methods:
        raise ValueError(f"Invalid transform_method: {transform_method}")

    if columns is None:
        columns = data.columns

    feature_types = {
        'numeric': data[columns].select_dtypes(include=['int64', 'float64']).columns,
        'categorical': data[columns].select_dtypes(include=['object']).columns,
        'binary': [col for col in columns if data[col].nunique() == 2]
    }

    n_cols = min(3, sum(len(features) for features in feature_types.values()))
    fig, axs = plt.subplots((sum(len(features) for features in feature_types.values()) - 1) // n_cols + 1, n_cols, figsize=(15, 5 * ((sum(len(features) for features in feature_types.values()) - 1) // n_cols + 1)))
    axs = axs.ravel()

    i = 0
    for ftype, features in feature_types.items():
        for feature in features:
            if ftype == 'numeric':
                if transform_method:
                    data[feature] = transform_methods[transform_method](data[feature])
                sns.histplot(data[feature], bins=bins, kde=True, ax=axs[i])
            else:
                top_categories = data[feature].value_counts().index[:max_categories]
                data.loc[data[feature].isin(top_categories), feature].value_counts().plot(kind='bar', ax=axs[i], color='steelblue')
            axs[i].set_title(f'Распределение признака {feature}')
            i += 1

    for j in range(i, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    return data

def calculate_and_add_percentages(data, columns=None, target=None, id_column=None):
    """
    Функция для расчета и добавления вероятности целевого действия для каждого уникального значения в указанных столбцах.

    Параметры:
        data (pd.DataFrame): Входной DataFrame.
        columns (list): Список имен столбцов для выполнения операции. Если не указан, используются все столбцы.
        target (str): Имя целевого столбца для расчета процента. По умолчанию - последний столбец dataframe.
        id_column (str): Уникальный идентификатор для группировки. Если не указан, используется первый столбец.

    Возвращает:
        data (pd.DataFrame): DataFrame с новыми столбцами, содержащими информацию о вероятности.
        percentage_dicts (dict): Словарь, содержащий словари вероятностей для каждого столбца.
    """
    data = data.copy()
    columns = columns or data.columns.tolist()

    id_column = id_column or data.columns[0]
    target = target or data.columns[-1]

    percentage_dicts = {}

    for column in columns:
        # Подсчитываем количество всех событий и положительных событий для каждого уникального значения в столбце
        total_counts = data.groupby(column)[target].count()
        positive_counts = data.groupby(column)[target].sum()

        # Расчитываем вероятности
        percentages = (positive_counts / total_counts) * 100

        # Добавляем новый столбец с вероятностями в исходный DataFrame
        data[f'{column}_probability'] = data[column].map(percentages.to_dict()).fillna(0)

        # Сохраняем словарь вероятностей для каждого столбца
        percentage_dicts[column] = percentages.to_dict()

    return data, percentage_dicts
    
def plot_correlation_matrix(data, threshold=0.8):
    """
    Plot a correlation matrix for the given DataFrame. 
    If two features have a correlation higher than the threshold, drop the second feature.
    Parameters:
        data: pandas DataFrame.
        threshold: float, optional. 
            Features with a correlation higher than this value are considered highly correlated. 
            The second feature of each pair of highly correlated features will be dropped.
            Defaults to 0.8.
    Returns:
        new_data: a new DataFrame with highly correlated features dropped. 
                  If no such features are found, returns None.
    """
    # Calculate correlation matrix
    corr = data.corr().round(2)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Use a custom diverging colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.show()

    # Identify pairs of features that are highly correlated
    highly_correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                featurename = corr.columns[i]
                highly_correlated_features.add(featurename)

    # Create a new DataFrame excluding highly correlated features
    if highly_correlated_features:
        new_data = data.drop(columns=highly_correlated_features)
        return new_data


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


class ProbabilityEncoder:
    def __init__(self, columns=None, id_column=None):
        self.columns = columns
        self.id_column = id_column
        self.percentage_dicts = {}

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.columns = self.columns if self.columns else data.columns.tolist()
        self.id_column = self.id_column if self.id_column else data.columns[0]
        self.target = data.columns[-1]

        for column in self.columns:
            total_counts = data.groupby(column)[self.target].count()
            positive_counts = data.groupby(column)[self.target].sum()
            percentages = (positive_counts / total_counts) * 100
            self.percentage_dicts[column] = percentages.to_dict()

        return self

    def transform(self, X):
        data = X.copy()
        for column, percentages in self.percentage_dicts.items():
            data[column] = data[column].map(percentages).fillna(0)

        return data

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
class DataPreprocessor:
    """
    Класс DataPreprocessor осуществляет предобработку данных, автоматически определяя числовые и категориальные признаки.

    Он применяет StandardScaler или MinMaxScaler к числовым признакам и OneHotEncoder к категориальным. Предобработка данных
    включает в себя обучение препроцессора на данных (метод fit), применение препроцессора к данным (метод transform) и
    комбинированный метод обучения и преобразования (метод fit_transform).
    """
    def __init__(self, scaling='scaler', encoding='onehot'):
        if scaling == 'scaler':
            self.num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        elif scaling == 'MinMax':
            self.num_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

        if encoding == 'label':
            self.cat_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])
        elif encoding == 'onehot':
            self.cat_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = None

    def fit(self, X):
        numeric_features = X.select_dtypes(include=['int', 'float']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, numeric_features),
                ('cat', self.cat_transformer, categorical_features)
            ])

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ModelTrainer:
    def __init__(self, task_type='regression', balance=False, models=None, n_jobs=-1):
        self.task_type = task_type
        self.balance = balance
        self.n_jobs = n_jobs
        self.models = self._get_models(models)
        self.trained_models = {}
        self.score_func = roc_auc_score if task_type == 'classification' else mean_squared_error
        if balance and task_type == 'classification':
            self.sampler = SMOTE()

    def _get_models(self, models):
        regression_models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=self.n_jobs),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'CatBoostRegressor': CatBoostRegressor(silent=True)
        }

        classification_models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=self.n_jobs),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'CatBoostClassifier': CatBoostClassifier(silent=True),
            'LGBMClassifier': LGBMClassifier(n_jobs=self.n_jobs)
        }

        models_dict = classification_models if self.task_type == 'classification' else regression_models
        models = models_dict.keys() if models is None else [model for model in models if model in models_dict]
        return [models_dict[model_name] for model_name in models]

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        best_model, best_score = None, -float('inf')
        for model in self.models:
            try:
                if self.balance and self.task_type == 'classification':
                    X_train, y_train = self._apply_balance_strategy(model, X_train, y_train)
                model.fit(X_train, y_train)
                score = self._evaluate(model, X_test, y_test)
                self.trained_models[type(model).__name__] = model
                if score > best_score:
                    best_model, best_score = model, score
            except NotFittedError as e:
                print(f"Model {type(model).__name__} could not be fitted. Error: {str(e)}")
        print(f"\nBest model: {type(best_model).__name__}, with Score: {best_score}")
        return best_model, self.trained_models

    def _apply_balance_strategy(self, model, X_train, y_train):
        balance_strategies = {
            LogisticRegression: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            RandomForestClassifier: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            LGBMClassifier: lambda x, y: (x, y, {'class_weight': 'balanced'}),
            GradientBoostingClassifier: lambda x, y: (*self.sampler.fit_resample(x, y), {}),
            CatBoostClassifier: lambda x, y: (*self.sampler.fit_resample(x, y), {})
        }
        strategy = balance_strategies.get(type(model), lambda x, y: (x, y, {}))
        X_train, y_train, params = strategy(X_train, y_train)
        model.set_params(**params)
        return X_train, y_train

    def _evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        model_name = type(model).__name__
        score = self.score_func(y_test, y_pred)
        metric_name = 'ROC_AUC' if self.task_type == 'classification' else 'MSE'
        print(f"Model: {model_name}, {metric_name}: {score}")
        return score

class ModelOptimizer:
    def __init__(self, models=None, cv=5, scorer='roc_auc', balance=False, n_jobs=-1):
        self.cv = cv
        self.balance = balance
        self.n_jobs = n_jobs
        self.models = self._get_models(models)
        self.param_grids = self._get_param_grids()
        self.scorer = get_scorer(scorer)

    def _get_models(self, models):
        model_classes = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=self.n_jobs),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'CatBoostRegressor': CatBoostRegressor(),
            'LogisticRegression': LogisticRegression(class_weight='balanced' if self.balance else None, n_jobs=self.n_jobs),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=self.n_jobs),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'CatBoostClassifier': CatBoostClassifier(),
            'LGBMClassifier': LGBMClassifier()
        }
        return [model_classes[model] for model in models] if models else [model for model in model_classes.values()]

    def _get_param_grids(self):
        return {
            'RandomForestRegressor': {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            },
            'GradientBoostingRegressor': {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "subsample": [0.5, 0.7, 1.0],
                "max_depth": [3, 7, 9]
            },
            'CatBoostRegressor': {
                "iterations": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "depth": [6, 8, 10]
            },
            'LogisticRegression': {
                "C": [0.01, 0.1, 1.0],
                "penalty": ['l1', 'l2']
            },
            'RandomForestClassifier': {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "subsample": [0.5, 0.7, 1.0],
                "max_depth": [3, 7, 9]
            },
            'CatBoostClassifier': {
                "iterations": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "depth": [6, 8, 10]
            },
            'LGBMClassifier': {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "max_depth": [3, 5, 7]}
        }

    def _is_classification(self):
        return any(isinstance(model, (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, CatBoostClassifier)) for model in self.models)

    def optimize(self, X_train, y_train):
        self.optimized_models = []  # Создаем пустой список для хранения оптимизированных моделей
        best_score = -1

        for model in self.models:
            model_name = type(model).__name__
            print(f"Optimizing {model_name}...")
            param_grid = self.param_grids.get(model_name, {})
            grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scorer, n_jobs=self.n_jobs)
            try:
                grid_search.fit(X_train, y_train)
            except Exception as e:
                print(f"An error occurred while fitting {model_name}: {str(e)}")
                continue
            params = grid_search.best_params_
            score = grid_search.best_score_
            print(f"Best parameters for {model_name}: {params}")
            print(f"Best score for {model_name}: {score}")

            # Добавляем оптимизированную модель и результаты в список
            self.optimized_models.append({"model": grid_search.best_estimator_, "params": params, "score": score})

            if score > best_score:
                self.best_model = grid_search.best_estimator_
                best_score = score

        return self.optimized_models


    def evaluate(self, X_test, y_test, models):
        for model in models:
            y_pred = model.predict(X_test)
            score = self.scorer(y_test, y_pred if self._is_classification() else y_pred[:, 1])
            print(f"Test score for {type(model).__name__}: {score}\n")