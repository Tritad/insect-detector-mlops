# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

client = TestClient(app)

def create_test_image():
    """สร้างข้อมูลรูปภาพจริงขนาดเล็กเพื่อใช้ทดสอบ"""
    file = io.BytesIO()
    image = Image.new('RGB', (224, 224), color = 'red')
    image.save(file, 'jpeg')
    file.seek(0)
    return file

def test_predict_success():
    """เช็คว่า API Endpoint /predict ทำงานได้และตอบกลับเป็น JSON ที่ถูกต้อง [cite: 91]"""
    # 1. เตรียมรูปภาพจริง
    test_img = create_test_image()
    file = {"file": ("test.jpg", test_img, "image/jpeg")}
    
    # 2. ยิง API
    response = client.post("/predict", files=file)
    
    # 3. ตรวจสอบผลลัพธ์
    assert response.status_code == 200
    data = response.json()
    assert "prediction_class" in data
    assert "confidence" in data
    assert data["status"] == "success"

def test_predict_not_image():
    """เช็คว่าระบบดักจับไฟล์ที่ไม่ใช่รูปภาพได้ถูกต้อง (Status 400) """
    file = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=file)
    
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_corrupted_image():
    """เช็คว่าระบบดักจับไฟล์รูปภาพที่เสียหายได้"""
    file = {"file": ("broken.jpg", b"not-a-real-image", "image/jpeg")}
    response = client.post("/predict", files=file)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_file_too_large():
    """เช็คว่าระบบบล็อกไฟล์ที่เกิน 5MB"""
    large_content = b"x" * (5 * 1024 * 1024 + 1)
    file = {"file": ("large.jpg", large_content, "image/jpeg")}
    response = client.post("/predict", files=file)

    assert response.status_code == 400
    assert "detail" in response.json()