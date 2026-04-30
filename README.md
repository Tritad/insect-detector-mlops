---
title: insect-detector-demo
sdk: docker
app_port: 7860
---

# Insect Detector MLOps

โปรเจกต์นี้เป็นระบบจำแนกภาพแมลงแบบ end-to-end สำหรับงาน MLOps โดยแยกเป็น API และ UI คนละ Space บน Hugging Face พร้อม CI/CD, ONNX inference, และชุดทดสอบประสิทธิภาพด้วย JMeter

## ภาพรวมระบบ

- API service: FastAPI สำหรับรับภาพและส่งผลทำนายที่ `/predict`
- UI service: Streamlit สำหรับอัปโหลดภาพและแสดงผลแบบอ่านง่าย
- Model runtime: ONNX + onnxruntime เพื่อให้รันบน CPU ได้เบาและเร็ว
- Deployment: GitHub Actions deploy อัตโนมัติไปยัง Hugging Face Spaces
- Performance test: JMeter สำหรับทดสอบ throughput, latency และ error rate

## สิ่งที่ทำได้

- อัปโหลดภาพแมลงหลายไฟล์พร้อมกัน
- ทำนายชนิดแมลงด้วยโมเดลที่ fine-tune มาแล้ว
- แสดงชื่อแมลงทั้งภาษาไทยและภาษาอังกฤษ
- แสดง confidence, อาการทำลาย, และคำแนะนำสารออกฤทธิ์เบื้องต้น
- มี health check และ root endpoint สำหรับตรวจสถานะบริการ

## สถานะล่าสุด

- API Space: https://mhrt03-insect-detector-demo.hf.space
- UI Space: https://mhrt03-insect-detector-ui.hf.space
- CI/CD: รันทดสอบก่อน deploy อัตโนมัติบน branch `main`
- Model format: ONNX สำหรับใช้งานจริงบน Hugging Face Spaces

## โครงสร้างโปรเจกต์

- `app/` - FastAPI backend
- `ui/` - Streamlit frontend
- `scripts/` - สคริปต์ export / optimize / train
- `tests/` - unit tests และ performance artifacts
- `.github/workflows/` - GitHub Actions pipeline
- `model/` - ไฟล์โมเดลและ preprocessor configuration

## รันโปรเจกต์แบบ Local

### ใช้ Docker Compose
```bash
docker compose up --build
```

### URLs ที่ใช้ทดสอบ
- API: http://localhost:8000
- UI: http://localhost:8501

### ทดลองเรียก API ด้วย cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

## การพัฒนาและปรับแต่งโมเดล

### Export เป็น ONNX
```bash
python scripts/export_onnx.py
```

### Optimize และ benchmark
```bash
python scripts/optimize.py
```

สคริปต์นี้จะสรุปผลเปรียบเทียบ เช่น:

- Original model latency และขนาดไฟล์
- ONNX latency และขนาดไฟล์
- Quantized ONNX latency และขนาดไฟล์

ผลลัพธ์สำหรับทำรายงานจะถูกบันทึกไว้ที่ `reports/optimization_metrics.csv`

## CI/CD

Workflow หลักอยู่ที่ `.github/workflows/ci-cd.yml`

พฤติกรรมของ pipeline:

- รัน `pytest` ทุกครั้งที่ push หรือเปิด pull request
- ถ้าทดสอบผ่านและ push ไปที่ `main` จะ deploy ไปยัง Hugging Face Spaces อัตโนมัติ
- แยก deploy เป็น 2 ส่วน คือ API Space และ UI Space

Repository secrets ที่ต้องมี:

- `HF_TOKEN` - Hugging Face access token ที่มีสิทธิ์เขียน
- `HF_SPACE_REPO` - repo ของ API Space ในรูปแบบ `username/space_name`
- `HF_UI_SPACE_REPO` - repo ของ UI Space ในรูปแบบ `username/space_name`

## Performance Testing

Artifacts สำหรับทดสอบโหลดอยู่ที่:

- JMeter plan: `tests/performance/insect_api_loadtest.jmx`
- ผลทดสอบ: `tests/performance/`

ใช้ dashboard ของ JMeter เพื่อดูค่า:

- Throughput (TPS)
- Latency (P95)
- Error rate

## หมายเหตุการใช้งาน

- UI จะเรียก API Space เพื่อขอผลทำนาย
- ถ้าไม่มี `st.secrets` ระบบ UI จะ fallback ไปยังค่าเริ่มต้นที่กำหนดไว้
- ควรใช้ภาพแมลงที่มีความคมชัดพอสมควรเพื่อให้ผลทำนายแม่นขึ้น

## ไฟล์สำคัญ

- `app/main.py` - FastAPI inference service
- `ui/app.py` - Streamlit interface
- `requirements-space.txt` - dependencies สำหรับ API Space
- `requirements-ui-space.txt` - dependencies สำหรับ UI Space
- `tests/test_api.py` - unit tests ของ API

## License / Usage

โปรเจกต์นี้จัดทำขึ้นเพื่อการศึกษาและงานวิจัยด้าน MLOps / image classification เป็นหลัก หากจะนำไปใช้งานจริง ควรทดสอบกับข้อมูลจริงและตรวจสอบการเลือกสารเคมีตามข้อกำหนดท้องถิ่นก่อนเสมอ
