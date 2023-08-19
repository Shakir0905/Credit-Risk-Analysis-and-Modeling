from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict, List, Union
from config import MODEL_PATH

app = FastAPI()
model = joblib.load(MODEL_PATH)

class Item(BaseModel):
    id: int
    data: Dict[str, Optional[Union[int, str, float]]]

@app.post("/predict")
def predict(items: List[Item]):
    data = pd.DataFrame([item.data for item in items])
    predictions = model.predict(data)
    return [{"client_id": item.id, "prediction": int(prediction)} for item, prediction in zip(items, predictions)]
