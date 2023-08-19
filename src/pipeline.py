from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
import logging
import preprocess
from config import DATA_PATH, TARGET_PATH
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)

def main():
    try:
        df = preprocess.process_parquet_dataset_from_zip(DATA_PATH)
        # df = pd.read_parquet(RESULT_DATA_PATH)
        df_aggregated = preprocess.process_and_save_chunks(df, chunk_ratio=0.1)
        # df_aggregated = pd.read_csv(DATA_ML_PATH)
        df_aggregated = df_aggregated.merge(pd.read_csv(TARGET_PATH), on='id') 

        logging.info("Data prepared successfully.")

        # Параметры для обучения
        params = {
            'num_leaves': 31, 
            'objective': 'binary', 
            'metric': 'auc', 
            'n_estimators': 1000,
            'class_weight': 'balanced'
        }

        X = df_aggregated.drop(['id', 'flag'], axis=1)
        y = df_aggregated['flag']
        
        # 100k примеров для теста
        X_remain, X_test, y_remain, y_test = train_test_split(
            X, y, test_size=100000, stratify=y, random_state=42
        )

        # разделим оставшиеся данные между тренировочным и валидационным наборами
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_remain, y_remain, test_size=0.2, stratify=y_remain, random_state=42
        )

        logging.info("Training the model ...")

        model = LGBMClassifier(verbose=-1, early_stopping_rounds=10, **params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        
        logging.info("Model trained successfully.")
        
        preds_test = model.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(y_test, preds_test)
        print(f"Test ROC-AUC Score: {roc_auc_test:.4f}")

        # Сохранение модели
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        logging.info(f"Model saved at {model_path}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
