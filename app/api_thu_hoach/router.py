from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import joblib

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Load models
model_thu_hoach = joblib.load("app/api_thu_hoach/model_forest_thu_hoach.pkl")
model_thuc_an = joblib.load("app/api_thu_hoach/model_forest_thuc_an.pkl")
model_tile_haohut = joblib.load("app/api_thu_hoach/model_forest_tilehaohut.pkl")

class HarvestInput(BaseModel):
    tuoi_hientai: int
    tong_food: int
    soluongca: int
    dien_tich: float
    tile_haohut: float
    target_size: float
    bi_benh: int
    gtm: int = 0
    tgtm: int = 0
    xh: int = 0
    xx: int = 0
    size_giong: int = 0
    
def growth_rate_by_age(age):
    if age <= 30:
        return 1.23
    elif age <= 60:
        return 2.04
    elif age <= 90:
        return 4.43
    elif age <= 120:
        return 5.19
    elif age <= 150:
        return 5.59
    elif age <= 180:
        return 5.84
    elif age <= 210:
        return 5.98
    elif age <= 240:
        return 7.62
    elif age <= 270:
        return 10.00
    elif age <= 300:
        return 8.28
    else:
        return 6.14  # Giá trị của nhóm '301-307' (có thể chỉnh sửa nếu cần)

@router.post("/predict_thu_hoach")
def predict_harvest(data: HarvestInput):
    if data.dien_tich <= 0:
        return {"error": "Diện tích ao phải lớn hơn 0."}
    if data.soluongca <= 0:
        return {"error": "Số lượng cá phải lớn hơn 0."}

    if data.bi_benh == 0:
        ck, gtm, tgtm, xh, xx = 1, 0, 0, 0, 0
    else:
        ck = 0
        gtm, tgtm, xh, xx = data.gtm, data.tgtm, data.xh, data.xx

    # Dự đoán size hiện tại
    temp_soluongca = data.soluongca
    temp_matdo = temp_soluongca / data.dien_tich
    thu_hoach_input = [[
        data.tuoi_hientai,
        data.tong_food,
        temp_matdo,
        data.soluongca,data.size_giong,
        ck, gtm, tgtm, xh, xx
    ]]
    predicted_size = model_thu_hoach.predict(thu_hoach_input)[0]
    current_size = predicted_size
    days_needed = 0
    tong_food = data.tong_food
    tile_haohut_start = 0

    while current_size < data.target_size and days_needed < 400:
        days_needed += 1
        new_tuoi = data.tuoi_hientai + days_needed

        # 1. Dự đoán lượng thức ăn hôm nay
        thuc_an_input = [[
            temp_soluongca * current_size / 1000, # sản lượng cá ngày này
            new_tuoi,
            tile_haohut_start + 0.1 * days_needed, # tỉ lệ hao hụt lũy kế đến ngày này
            temp_matdo
        ]]
        food_today = model_thuc_an.predict(thuc_an_input)[0]

        # 2. Cộng vào tổng food
        tong_food += food_today
        if isinstance(tong_food, (np.ndarray, list)):
            tong_food = float(tong_food[-1])  # hoặc tong_food[0] nếu đúng ý bạn
        else:
            tong_food = float(tong_food)

        # 3. Tăng tỉ lệ hao hụt lũy kế thêm 0.1% mỗi ngày
        tile_haohut_luyke = tile_haohut_start + 0.1 * days_needed

        # 4. Số lượng cá còn lại (dùng lũy kế)
        temp_soluongca = data.soluongca * (1 - tile_haohut_luyke / 100)
        size_giong = data.size_giong
        # 5. Tính lại mật độ
        temp_matdo = temp_soluongca / data.dien_tich

        # 6. Dự đoán size mới
        print("new_tuoi:", type(new_tuoi), new_tuoi)
        print("tong_food:", type(tong_food), tong_food)
        print("temp_matdo:", type(temp_matdo), temp_matdo)
        print("tile_haohut_luyke:", type(tile_haohut_luyke), tile_haohut_luyke)
        print("size:", type(current_size), current_size)
        updated_features = [[
    float(new_tuoi), float(tong_food), float(temp_matdo), float(tile_haohut_luyke),size_giong,
    int(ck), int(gtm), int(tgtm), int(xh), int(xx)
]]

        new_size = model_thu_hoach.predict(updated_features)[0]
        max_growth = growth_rate_by_age(new_tuoi)
        delta = new_size - current_size

        if delta > max_growth:
            new_size = current_size + max_growth  # giới hạn tăng trưởng
        elif delta < 0:
            new_size = current_size  # không giảm size

        # 8. Cập nhật size hiện tại
        alpha = 0.7
        current_size = alpha * new_size + (1 - alpha) * current_size

        


    # Sản lượng cuối cùng
    final_sanluong = temp_soluongca * current_size / 1000

    if days_needed < 400:
        return {
            "size_hientai": round(predicted_size, 1),
            "target_size": data.target_size,
            "days_can_dat_target": days_needed,
            "matdo_cuoicung": round(temp_matdo, 2),
            "soluongca_cuoicung": int(temp_soluongca),
            "sanluong_thu_hoach": round(final_sanluong, 2)
        }
    else:
        return {
            "size_hientai": round(predicted_size, 1),
            "target_size": data.target_size,
            "days_can_dat_target": None,
            "message": "Không thể dự đoán đạt size mục tiêu trong 400 ngày."
        }
