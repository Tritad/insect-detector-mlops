FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependencies ที่จำเป็นสำหรับ ONNX และ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ด API และโมเดล
COPY ./app ./app
COPY ./model/onnx_v2 ./model/onnx_v2

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]