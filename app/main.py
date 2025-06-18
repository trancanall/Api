from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Import các router
from app.api_thuc_an.router import router as thuc_an_router
from app.api_thay_nuoc.router import router as thaynuoc_router
from app.api_dich_benh.router import router as dichbenh_router
from app.api_thu_hoach.router import router as thuhoach_router
from app.api_cap_thuoc.router import router as capthuoc_router
from app.api_thu_hoach_lui.router import router as thu_hoach_lui_router

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Giao diện trang chủ
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Mount routers
app.include_router(thuc_an_router, prefix="/thucan", tags=["Thức Ăn"])
app.include_router(thaynuoc_router, prefix="/thaynuoc", tags=["Thay Nước"])
app.include_router(dichbenh_router, prefix="/dichbenh", tags=["Dịch bệnh"])
app.include_router(thuhoach_router, prefix="/thuhoach", tags=["Thu hoạch"])
app.include_router(capthuoc_router, prefix="/capthuoc", tags=["Cấp thuốc"])
app.include_router(thu_hoach_lui_router, prefix="/thuhoachlui", tags=["Thu hoạch lùi"])
