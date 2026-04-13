---
title: insect-detector-demo
sdk: docker
app_port: 7860
---

# High-Throughput Image Classification Service (MLOps)

This project provides a production-oriented FastAPI service for high-throughput insect image classification using MobileNetV2 (fine-tuned), ONNX, and optimized CPU inference.

## Key Features
- FastAPI async endpoint at `/predict`
- CPU-bound inference executed with `ProcessPoolExecutor`
- Input validation and error handling
- ONNX export and dynamic quantization scripts
- Dockerized backend and frontend
- CI/CD workflow for test and auto-deploy to Hugging Face Spaces

## Project Structure
- `app/`: FastAPI backend
- `ui/`: Streamlit frontend
- `scripts/`: training/export/optimization scripts
- `tests/`: API tests and performance artifacts
- `.github/workflows/`: CI/CD pipeline

## Run Locally (Docker)
```bash
docker compose up --build
```

Backend URL: `http://localhost:8000`
Frontend URL: `http://localhost:8501`

## API Example (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

## Optimization Workflow
1. Export ONNX model:
```bash
python scripts/export_onnx.py
```
2. Run optimization benchmark and dynamic quantization:
```bash
python scripts/optimize.py
```

The script prints a comparison table for:
- Original model latency and size
- ONNX latency and size
- Quantized ONNX latency and size

It also saves report-ready metrics to: `reports/optimization_metrics.csv`

## CI/CD
Workflow file: `.github/workflows/ci-cd.yml`

Pipeline behavior:
- Run `pytest` on push and pull request
- If tests pass and push is on `main`, auto-deploy to Hugging Face Spaces

Required repository secrets:
- `HF_TOKEN`: Hugging Face access token (write permission)
- `HF_SPACE_REPO`: Space path in the format `username/space_name`

## Performance Testing Artifacts
- JMeter plan: `tests/performance/insect_api_loadtest.jmx`
- Postman collection: `tests/performance/postman_collection.json`

Use JMeter dashboard output to report Throughput (TPS) and Latency (P95).
