from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import joblib
import pandas as pd
router = APIRouter()

# Load model
model = joblib.load("app/api_thu_hoach_lui/model_forest_thu_hoach_lui.pkl")
feature_cols = [
    'tong_food', 'matdo', 'tile_hao hut', 'target_size',
    'size_giong', 'size_interp', 'ck', 'gtm', 'xh', 'xx'
]

# Input schema
class HarvestInput(BaseModel):
    tong_food: float
    matdo: float
    tile_hao_hut: float
    size_giong: float
    size_interp: float
    ck: int = 0
    gtm: int = 0
    xh: int = 0
    xx: int = 0
    target_size: float = 1400  # default


@router.post("/predict_thu_hoach_lui/")
def predict_harvest_delay(data: HarvestInput):
    try:
        user_input = data.dict()
        input_ordered = [user_input.get(col.replace(" ", "_")) for col in feature_cols]
        input_df = pd.DataFrame([input_ordered], columns=feature_cols)

        prediction = model.predict(input_df)[0]

        return {
            "du_doan_so_ngay_con_lai": int(prediction),
            "thong_tin_dau_vao": user_input
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
