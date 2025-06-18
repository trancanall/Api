from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import APIRouter, Request

# Load mô hình

# Khởi tạo app
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
model_forest_dich_benh = joblib.load("app/api_dich_benh/model_forest_dich_benh.pkl")

# Mô tả các nhãn bệnh
labels = ['gtm', 'tgtm', 'xh']
descriptions = {
    'gtm': 'Gan thận mủ',
    'tgtm': 'Trắng gan trắng mang',
    'xh': 'Xuất huyết',
}

# Khai báo dữ liệu đầu vào
class InputFeatures(BaseModel):
    tuoi: float
    matdo: float
    sanluongca: float
    tile_hao_hut: float
    size: float
    soluongca: int

# Endpoint dự đoán
@router.post("/predict_dich_benh")
def du_doan_benh(features: InputFeatures):
    # Tạo mảng đầu vào
    new_data = np.array([[features.tuoi, features.matdo, features.sanluongca,
                          features.tile_hao_hut, features.size, features.soluongca]])

    # Dự đoán
    prediction = model_forest_dich_benh.predict(new_data)
    predicted_labels = prediction[0]

    # Lấy tên bệnh
    tinh_trang_list = [descriptions[labels[i]] for i in range(len(labels)) if predicted_labels[i] == 1]

    # Trả kết quả
    if tinh_trang_list:
        return {"ket_qua": tinh_trang_list}
    else:
        return {"ket_qua": ["Cá chỉ xây xát hoặc bình thường."]}
