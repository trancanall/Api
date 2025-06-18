from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import APIRouter, Request

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Load models
model_forest_phanloai = joblib.load("app/api_thay_nuoc/model_forest_phanloai.pkl")
model_forest_biendo = joblib.load("app/api_thay_nuoc/model_forest_biendo.pkl")
model_forest_biendo2 = joblib.load("app/api_thay_nuoc/model_forest_biendo2.pkl")
model_gio = joblib.load("app/api_thay_nuoc/model_gio.pkl")

# Input schema
class FishInput(BaseModel):
    size: int
    thuocnuoc: float
    tuoi: int
    tong_food: int
    bi_benh: int  # 0 hoặc 1
    xx: int = 0
    xh: int = 0
    gtm: int = 0
    tgtm: int = 0
    loaithucan: int = 0  # default nếu không có chọn


@router.post("/predict/")
def predict_water_change(data: FishInput):
    if data.bi_benh == 0:
        ck = 1
        xx = xh = gtm = tgtm = 0
    else:
        ck = 0
        xx = data.xx
        xh = data.xh
        gtm = data.gtm
        tgtm = data.tgtm

    # Model phân loại
    X_input = np.array([[data.size, data.tuoi, data.tong_food, ck, gtm, tgtm, xh, xx]])
    prediction = model_forest_phanloai.predict(X_input)[0]

    label_map = {0: "Không thay nước", 1: "Thay nước đợt 1", 2: "Thay nước 2 đợt"}
    result = {
        "phan_loai": label_map.get(prediction),
        "bien_do": [],
        "ty_le_thay": []
    }

    def du_doan_biendo(model, features, dot=""):
        biendo = model.predict(np.array([features]))[0]
        if data.thuocnuoc != 0:
            ty_le = round(100 * biendo / data.thuocnuoc, 2)
        else:
            ty_le = None
        return {
            "dot": dot or "đợt 1",
            "bien_do_cm": round(biendo, 2),
            "ty_le_thay": ty_le
        }
    
    def du_doan_buoi_thay(size, tuoi, tong_food, ck, gtm, tgtm, xh, xx, loaithucan):
        features = np.array([[size, tuoi, loaithucan, tong_food, ck, gtm, tgtm, xh, xx]])
        gio_pred = model_gio.predict(features)[0]

        label_map = {
            0: "chiều",
            1: "sáng",
            2: "tối",
            3: "trưa"
        }
        return label_map.get(gio_pred, "không xác định")




    if prediction == 1:
        features = [data.size, data.thuocnuoc, data.tuoi, ck, gtm, tgtm, xh, xx]
        result["bien_do"].append(du_doan_biendo(model_forest_biendo, features))
        gio = du_doan_buoi_thay(data.size, data.tuoi, data.tong_food, ck, gtm, tgtm, xh, xx, data.loaithucan)
        result["gio"] = [f"Thay nước buổi {gio}"]

    elif prediction == 2:
        features1 = [data.size, data.thuocnuoc, data.tuoi, ck, gtm, tgtm, xh, xx]
        features2 = [data.size, data.thuocnuoc, data.tuoi, xh, xx]
        result["bien_do"].append(du_doan_biendo(model_forest_biendo, features1, dot="đợt 1"))
        result["bien_do"].append(du_doan_biendo(model_forest_biendo2, features2, dot="đợt 2"))
        # Đợt 1: dự đoán
        gio1 = du_doan_buoi_thay(data.size, data.tuoi, data.tong_food, ck, gtm, tgtm, xh, xx, data.loaithucan)
        if gio1 == 2:
            result["gio"] = [f"Đợt 1: buổi sáng", "Đợt 2: buổi tối"]
        else:
            result["gio"] = [f"Đợt 1: buổi {gio1}", "Đợt 2: buổi tối"]
    return result
