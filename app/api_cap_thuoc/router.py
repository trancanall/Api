from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

router = APIRouter()

# Load models and assets
features_cols = joblib.load("app/api_cap_thuoc/features_cols.pkl")
drug_classifier = joblib.load("app/api_cap_thuoc/drug_classifier_lgbm.pkl")
optimal_thresholds = joblib.load("app/api_cap_thuoc/optimal_thresholds_lgbm.pkl")
label_names = joblib.load("app/api_cap_thuoc/labels_to_keep.pkl")

# Rule maps
size_condition_map = {
    'antistress': lambda size: size < 200,
    'cefo': lambda size: size < 200,
    'hepasol': lambda size: size < 200,
    'levo': lambda size: size < 200,
    'liver red': lambda size: size < 200,
    'parasitol': lambda size: 50 <= size <= 200,
    'vimax': lambda size: size > 200,
}

size_note_map = {
    'antistress': "antistress chỉ thích hợp cho cá giai đoạn < 200gram",
    'cefo': "cefo chỉ thích hợp cho cá giai đoạn < 200gram",
    'hepasol': "hepasol chỉ thích hợp cho cá giai đoạn < 200gram",
    'levo': "levo chỉ thích hợp cho cá giai đoạn < 200gram",
    'liver red': "liver red chỉ thích hợp cho cá giai đoạn < 200gram",
    'parasitol': "parasitol chỉ dùng cho cá size 50–200gram",
    'vimax': "vimax chỉ dùng cho cá size > 200gram",
}

lieu_luong_map = {
    'antistress':      lambda sanluong: f"Gợi ý liều lượng: {sanluong/20000:.2f} – {sanluong/15000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/15-20 tấn cá/ngày)",
    'c40%':            lambda sanluong: f"Gợi ý liều lượng: {sanluong/50000:.2f} – {sanluong/40000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/40-50 tấn cá/ngày)",
    'cefo':            lambda sanluong: f"Gợi ý liều lượng: {sanluong/100000:.2f} – {sanluong/80000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/80-100 tấn cá/ngày)",
    'glucan':          lambda sanluong: f"Gợi ý liều lượng: {sanluong/20000:.2f} – {sanluong/15000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/15-20 tấn cá/ngày)",
    'hepasol':         lambda sanluong: f"Gợi ý liều lượng: {sanluong/20000:.2f} – {sanluong/15000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/15-20 tấn cá/ngày)",
    'levo':            lambda sanluong: f"Gợi ý liều lượng: {sanluong/100000:.2f} – {sanluong/80000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/80-100 tấn cá/ngày)",
    'liver red':       lambda sanluong: f"Gợi ý liều lượng: {sanluong/15000:.2f} – {sanluong/10000:.2f} lít/ngày cho {sanluong:,} kg cá (theo 1 lít/10-15 tấn cá/ngày)",
    'parasitol':       lambda sanluong: f"Gợi ý liều lượng: {sanluong/30000:.2f} – {sanluong/25000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/25-30 tấn cá/ngày)",
    'premix':          lambda sanluong: f"Gợi ý liều lượng: {sanluong/30000:.2f} – {sanluong/20000:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/20-30 tấn cá/ngày)",
    'prozyme':         lambda sanluong: f"Gợi ý liều lượng: {sanluong/1000:.2f} – {sanluong/500:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/500-1000 kg cá/ngày)",
    's.zyme':          lambda sanluong: f"Gợi ý liều lượng: {sanluong/1000:.2f} – {sanluong/500:.2f} kg/ngày cho {sanluong:,} kg cá (theo 1 kg/500-1000 kg cá/ngày)",
    'sorpherol':       lambda sanluong: f"{sanluong/20000:.2f} kg/20 tấn cá/ngày",
    'vimax':           lambda sanluong: f"Gợi ý liều lượng: {sanluong/45000:.2f} – {sanluong/40000:.2f} lít/ngày cho {sanluong:,} kg cá (theo 1 lít/40-45 tấn cá/ngày)",
    'yucca':           None
}


class FishInput(BaseModel):
    tuoi: int
    matdo: float
    tile_hao_hut: float
    loai_xl: int
    size: int
    xh: int
    xx: int
    ck: int
    gtm: int
    tgtm: int
    tong_food_ngay: float
    sanluong_kg: int
    the_tich: float = 0

def calculate_yucca_dosage(sanluong, thetich_nuoc):
    if thetich_nuoc <= 0:
        return "Vui lòng cung cấp thể tích nước hợp lệ"
    return f"Gợi ý liều lượng: {thetich_nuoc / 8000:.2f} lít (theo 1 lít/8000 m³ nước/ngày)"

@router.post("/predict_thuoc/")
def predict_drug(input_data: FishInput):
    try:
        user_dict = input_data.dict()
        sanluong = user_dict.pop("sanluong_kg")

        X_input = pd.DataFrame([user_dict], columns=features_cols)
        y_proba = drug_classifier.predict_proba(X_input)
        y_matrix = np.column_stack([p[:, 1] for p in y_proba])
        y_pred = (y_matrix >= optimal_thresholds).astype(int).flatten()

        # Gợi ý thuốc
        suggested = [label for label, val in zip(label_names, y_pred) if val == 1]
        
        # Liều lượng
        dosage_result = {}
        for label in suggested:
            if label in size_condition_map and not size_condition_map[label](user_dict["size"]):
                dosage_result[label] = size_note_map.get(label, "Không phù hợp size")
            elif label == 'yucca':
                thetich_nuoc = user_dict.get("the_tich", 0)
                dosage_result[label] = calculate_yucca_dosage(sanluong, thetich_nuoc)
            else:
                dosage_result[label] = lieu_luong_map.get(label, lambda _: "Không có rule liều lượng")(sanluong)

        # Chi tiết dự đoán
        detail_table = []
        for label, pred, prob, thresh in zip(label_names, y_pred, y_matrix.flatten(), optimal_thresholds):
            detail_table.append({
                "thuoc": label,
                "du_doan": int(pred),
                "xac_suat": round(float(prob), 4),
                "threshold": round(float(thresh), 4)
            })

        return {
            "thuoc_goi_y": suggested,
            "lieu_luong": dosage_result,
            "bang_chi_tiet": detail_table
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

