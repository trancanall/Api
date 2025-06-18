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
model_class = joblib.load("app/api_thuc_an/model_forest_class_new.pkl")
model_sang = joblib.load("app/api_thuc_an/model_forest_sang_lastest.pkl")
model_chieu = joblib.load("app/api_thuc_an/model_forest_chieu_new.pkl")
model_2buoi = joblib.load("app/api_thuc_an/model_xgb_multi.pkl")
model_loai_thuc_an = joblib.load("app/api_thuc_an/model_forest_loai_thuc_an.pkl")
label_encoders = joblib.load("app/api_thuc_an/label_encoders.pkl")
model_size = joblib.load("app/api_thuc_an/model_size.pkl")

# Danh sách tên thuốc chuẩn hóa từ model_class (chứa tất cả)
base_feature_names = list(model_class.feature_names_in_)
thuoc_list = [f for f in base_feature_names if f not in
              ['size_interp', 'sanluongca', 'tuoi', 'tile_hao hut', 'matdo', 'soluongca', 'loai_xl',
               'ck', 'gtm', 'tgtm', 'xh', 'xx']]

# Input schema
class FishFeedInput(BaseModel):
    size_interp: float
    sanluongca: float
    tuoi: int
    tile_hao_hut: float
    matdo: float
    soluongca: int
    loai_xl: int
    bi_benh: int  # 0 hoặc 1
    xx: int = 0
    xh: int = 0
    gtm: int = 0
    tgtm: int = 0
    thuoc: list[int]  # đúng độ dài len(thuoc_list)
    so_ngay_du_doan: int = 0  # số ngày muốn dự đoán liên tiếp
    dien_tich: float  # <-- Thêm dòng này
    size_giong: int = 0

@router.post("/predict_feed/")
def predict_feed(data: FishFeedInput):
    # 1. Kiểm tra đầu vào
    if data.tuoi < 3:
        return {"message": "Cá mới thả, không nên cho ăn."}
    if len(data.thuoc) != len(thuoc_list):
        return {"error": f"Danh sách thuốc phải có đúng {len(thuoc_list)} giá trị (0 hoặc 1)."}

    # 2. Xử lý trường hợp bệnh
    if data.bi_benh == 0:
        ck, xx, xh, gtm, tgtm = 1, 0, 0, 0, 0
    else:
        ck = 0
        xx = data.xx
        xh = data.xh
        gtm = data.gtm
        tgtm = data.tgtm

    def build_input_vector(model, state, thuoc, extra={}):
        vector = []
        for name in model.feature_names_in_:
            if name in state:
                vector.append(state[name])
            elif name in thuoc_list:
                idx = thuoc_list.index(name)
                vector.append(thuoc[idx])
            elif name in extra:
                vector.append(extra[name])
            else:
                vector.append(0)
        return np.array([vector])

    # ====== DỰ ĐOÁN 1 NGÀY ======
    if not data.so_ngay_du_doan or data.so_ngay_du_doan <= 1:
        state = {
            'size_interp': data.size_interp,
            'sanluongca': data.sanluongca,
            'tuoi': data.tuoi,
            'tile_hao_hut': data.tile_hao_hut,
            'matdo': data.matdo,
            'soluongca': data.soluongca,
            'loai_xl': data.loai_xl,
            'ck': ck,
            'gtm': gtm,
            'tgtm': tgtm,
            'xh': xh,
            'xx': xx,
            'dien_tich': data.dien_tich,
            'size_giong': data.size_giong
        }
        feature_vector = build_input_vector(model_class, state, data.thuoc)
        loai_cho_an = int(model_class.predict(feature_vector)[0])
        result = {"loai_cho_an": loai_cho_an}
        try:
            y_pred_new = model_loai_thuc_an.predict(feature_vector)
            predicted_label = label_encoders['loaithucan'].inverse_transform(y_pred_new)[0]
            result["loai_thuc_an"] = predicted_label
        except:
            result["loai_thuc_an"] = "Không xác định"

        # Dự đoán lượng cho ăn
        if loai_cho_an == 1:
            x_2buoi = build_input_vector(model_2buoi, state, data.thuoc)
            food_sang, food_chieu = map(float, model_2buoi.predict(x_2buoi)[0])
            result.update({
                "sang": round(food_sang, 2),
                "chieu": round(food_chieu, 2),
                "tong": round(food_sang + food_chieu, 2)
            })
        elif loai_cho_an == 2:
            x_sang = build_input_vector(model_sang, state, data.thuoc)
            food = model_sang.predict(x_sang)[0]
            result["sang"] = round(food, 2)
        elif loai_cho_an == 3:
            x_chieu = build_input_vector(model_chieu, state, data.thuoc)
            food = model_chieu.predict(x_chieu)[0]
            result["chieu"] = round(food, 2)
        return result

    # ====== DỰ ĐOÁN NHIỀU NGÀY ======
    du_doan_nhieu_ngay = []
    # Khởi tạo state ban đầu
    state = {
        'size_interp': data.size_interp,
        'sanluongca': data.sanluongca,
        'tuoi': data.tuoi,
        'tile_hao_hut': data.tile_hao_hut,
        'matdo': data.matdo,
        'soluongca': data.soluongca,
        'loai_xl': data.loai_xl,
        'ck': ck,
        'gtm': gtm,
        'tgtm': tgtm,
        'xh': xh,
        'xx': xx,
        'dien_tich': data.dien_tich,
        'size_giong': data.size_giong
    }
    thuoc = data.thuoc
    soluong_ban_dau = data.soluongca
    tile_hao_hut_start = state['tile_hao_hut']  # hoặc data.tile_hao_hut
    du_doan_nhieu_ngay = []
    for day in range(int(data.so_ngay_du_doan)):
        tile_hao_hut_luy_ke = tile_hao_hut_start + 0.1 * day
        next_soluongca = soluong_ban_dau * (1 - tile_hao_hut_luy_ke / 100)
        # 2. Số lượng cá giảm vật lý
        next_soluongca = next_soluongca * (1 - tile_hao_hut_luy_ke / 100)

    # 3. Size cá không giảm so với hôm qua
        max_increase = 1.43
        if "model_size" in globals():
            x_size = build_input_vector(model_size, state, thuoc)
            next_size_pred = float(model_size.predict(x_size)[0])
            # Tính mức tăng thực tế của model
            real_increase = next_size_pred - state['size_interp']
            # Nếu tăng vượt quá 1.43 thì chỉ tăng tối đa 1.43
            if real_increase > max_increase:
                next_size = state['size_interp'] + max_increase
            else:
                next_size = next_size_pred
        else:
            next_size = state['size_interp'] + max_increase

        # 4. Mật độ cá
        next_matdo = next_soluongca / state['dien_tich']
        next_sanluongca = next_soluongca * next_size / 1000

    # 6. Dự đoán loại cho ăn, loại thức ăn, lượng thức ăn...
        feature_vector = build_input_vector(
            model_class,
            {**state, "size_interp": next_size, "soluongca": next_soluongca,'size_giong':size_giong, "matdo": next_matdo, "sanluongca": next_sanluongca, "tile_hao_hut": tile_hao_hut_luy_ke},
            thuoc
        )
        loai_cho_an = int(model_class.predict(feature_vector)[0])
        try:
            y_pred_new = model_loai_thuc_an.predict(feature_vector)
            predicted_label = label_encoders['loaithucan'].inverse_transform(y_pred_new)[0]
        except:
            predicted_label = "Không xác định"

        # Lượng cho ăn (nếu muốn, giữ nguyên logic cũ)
        feed_info = {}
        if loai_cho_an == 1:
            x_2buoi = build_input_vector(
                model_2buoi,
                {**state, "size_interp": next_size, "soluongca": next_soluongca,'size_giong':size_giong, "matdo": next_matdo, "sanluongca": next_sanluongca, "tile_hao_hut": tile_hao_hut_luy_ke},
                thuoc
            )
            food_sang, food_chieu = map(float, model_2buoi.predict(x_2buoi)[0])
            feed_info = {
                "sang": round(food_sang, 2),
                "chieu": round(food_chieu, 2),
                "tong": round(food_sang + food_chieu, 2)
            }
        elif loai_cho_an == 2:
            x_sang = build_input_vector(
                model_sang,
                {**state, "size_interp": next_size, "soluongca": next_soluongca,'size_giong':size_giong, "matdo": next_matdo, "sanluongca": next_sanluongca, "tile_hao_hut": tile_hao_hut_luy_ke},
                thuoc
            )
            food = model_sang.predict(x_sang)[0]
            feed_info = {"sang": round(food, 2)}
        elif loai_cho_an == 3:
            x_chieu = build_input_vector(
                model_chieu,
                {**state, "size_interp": next_size, "soluongca": next_soluongca,'size_giong':size_giong, "matdo": next_matdo, "sanluongca": next_sanluongca, "tile_hao_hut": tile_hao_hut_luy_ke},
                thuoc
            )
            food = model_chieu.predict(x_chieu)[0]
            feed_info = {"chieu": round(food, 2)}

        du_doan_nhieu_ngay.append({
            "ngay": day + 1,
            "tuoi": state['tuoi'] + 1,
            "size_interp": round(next_size, 2),
            "soluongca": int(next_soluongca),
            "matdo": round(next_matdo, 2),
            "sanluongca": round(next_sanluongca, 2),
            "tile_hao_hut": round(tile_hao_hut_luy_ke, 2),
            "loai_cho_an": loai_cho_an,
            "loai_thuc_an": predicted_label,
            **feed_info
        })

        # Cập nhật state cho ngày tiếp theo
        state['tuoi'] += 1
        state['size_interp'] = next_size
        state['soluongca'] = next_soluongca
        state['matdo'] = next_matdo
        state['sanluongca'] = next_sanluongca
        state['tile_hao_hut'] = tile_hao_hut_luy_ke
        state['size_giong'] = size_giong
    return {"du_doan_nhieu_ngay": du_doan_nhieu_ngay}

    try:
        # Dự đoán nhãn
        y_pred_new = model_loai_thuc_an.predict(X_class)
        # Giải mã tên loại
        predicted_label = label_encoders['loaithucan'].inverse_transform(y_pred_new)[0]
        result["loai_thuc_an"] = predicted_label
    except Exception as e:
        result["loai_thuc_an"] = "Không xác định"

    # Hàm dựng feature theo đúng input model
    def build_input_vector(model, extra={}):
        vector = []
        for name in model.feature_names_in_:
            if name == 'size_interp':
                vector.append(data.size_interp)
            elif name == 'sanluongca':
                vector.append(data.sanluongca)
            elif name == 'tuoi':
                vector.append(data.tuoi)
            elif name == 'tile_hao_hut':
                vector.append(data.tile_hao_hut)
            elif name == 'size_giong':
                vector.append(data.size_giong)
            elif name == 'matdo':
                vector.append(data.matdo)
            elif name == 'soluongca':
                vector.append(data.soluongca)
            elif name == 'loai_xl':
                vector.append(data.loai_xl)
            elif name == 'ck':
                vector.append(ck)
            elif name == 'gtm':
                vector.append(gtm)
            elif name == 'tgtm':
                vector.append(tgtm)
            elif name == 'xh':
                vector.append(xh)
            elif name == 'xx':
                vector.append(xx)
            elif name in thuoc_list:
                idx = thuoc_list.index(name)
                vector.append(data.thuoc[idx])
            elif name in extra:
                vector.append(extra[name])
            else:
                vector.append(0)
        return np.array([vector])

    if loai_cho_an == 1:
        x_2buoi = build_input_vector(model_2buoi)
        food_sang, food_chieu = map(float, model_2buoi.predict(x_2buoi)[0])
        result.update({
            "sang": round(food_sang, 2),
            "chieu": round(food_chieu, 2),
            "tong": round(food_sang + food_chieu, 2)
        })

    elif loai_cho_an == 2:
        x_sang = build_input_vector(model_sang)
        food = model_sang.predict(x_sang)[0]
        result["sang"] = round(food, 2)

    elif loai_cho_an == 3:
        x_chieu = build_input_vector(model_chieu)
        food = model_chieu.predict(x_chieu)[0]
        result["chieu"] = round(food, 2)

    return result
@router.get("/get_thuoc_list")
def get_thuoc_list():
    return {"thuoc": thuoc_list}
