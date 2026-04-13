FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY ./app ./app
COPY ./model/onnx_v2 ./model/onnx_v2

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
