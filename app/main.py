import asyncio
import json
import io
import logging
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .schemas import PredictionResponse

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "onnx_v2"),
    os.path.join(BASE_DIR, "model", "onnx"),
]
DEFAULT_MODEL_ONNX_URL = (
    "https://raw.githubusercontent.com/Tritad/insect-detector-mlops/main/model/onnx_v2/model.onnx"
)


def _ensure_onnx_model_file():
    candidate_dir = next((path for path in MODEL_DIR_CANDIDATES if os.path.isdir(path)), MODEL_DIR_CANDIDATES[0])
    os.makedirs(candidate_dir, exist_ok=True)
    model_path = os.path.join(candidate_dir, "model.onnx")

    if os.path.exists(model_path):
        return candidate_dir

    model_url = os.getenv("MODEL_ONNX_URL", DEFAULT_MODEL_ONNX_URL)
    try:
        logger.info("model.onnx not found; downloading from %s", model_url)
        urllib.request.urlretrieve(model_url, model_path)
        return candidate_dir if os.path.exists(model_path) else None
    except Exception:
        logger.exception("Failed to download model.onnx from %s", model_url)
        return None


MODEL_DIR = next(
    (path for path in MODEL_DIR_CANDIDATES if os.path.exists(os.path.join(path, "model.onnx"))),
    None,
)

if MODEL_DIR is None:
    MODEL_DIR = _ensure_onnx_model_file()

if MODEL_DIR is None:
    raise RuntimeError("ไม่พบไฟล์โมเดล ONNX (model.onnx)")

MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

# Loaded lazily per worker process to avoid pickling heavyweight objects.
_worker_processor = None
_worker_session = None
_executor = None


def _get_worker_inference_components():
    global _worker_processor, _worker_session
    if _worker_processor is None:
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor_config.json")
        with open(preprocessor_path, "r", encoding="utf-8") as f:
            _worker_processor = json.load(f)
    if _worker_session is None:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        _worker_session = ort.InferenceSession(MODEL_PATH, sess_options=session_options)
    return _worker_processor, _worker_session


def _extract_target_size(preprocessor_cfg):
    crop_size = preprocessor_cfg.get("crop_size") or {}
    size_cfg = preprocessor_cfg.get("size") or {}

    if isinstance(crop_size, dict):
        height = int(crop_size.get("height") or crop_size.get("shortest_edge") or 224)
        width = int(crop_size.get("width") or crop_size.get("shortest_edge") or height)
        return width, height

    if isinstance(size_cfg, dict):
        height = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
        width = int(size_cfg.get("width") or size_cfg.get("shortest_edge") or height)
        return width, height

    if isinstance(size_cfg, int):
        return int(size_cfg), int(size_cfg)

    return 224, 224


def _preprocess_image(img: Image.Image, preprocessor_cfg):
    width, height = _extract_target_size(preprocessor_cfg)
    image = img.resize((width, height), Image.BILINEAR)
    pixel_values = np.asarray(image).astype(np.float32)

    if preprocessor_cfg.get("do_rescale", True):
        rescale_factor = float(preprocessor_cfg.get("rescale_factor", 1.0 / 255.0))
        pixel_values = pixel_values * rescale_factor

    if preprocessor_cfg.get("do_normalize", True):
        image_mean = np.array(preprocessor_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        image_std = np.array(preprocessor_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        pixel_values = (pixel_values - image_mean) / image_std

    # NHWC -> NCHW
    pixel_values = np.transpose(pixel_values, (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0).astype(np.float32)
    return pixel_values


def _get_executor():
    global _executor
    if _executor is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        _executor = ProcessPoolExecutor(max_workers=max_workers)
    return _executor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        global _executor
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=True)
            _executor = None


app = FastAPI(title="Pest/Insect Detector API", lifespan=lifespan)


def predict_sync(image_bytes: bytes):
    try:
        preprocessor_cfg, session = _get_worker_inference_components()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = _preprocess_image(img, preprocessor_cfg)
        ort_inputs = {session.get_inputs()[0].name: pixel_values}
        outputs = session.run(None, ort_inputs)

        logits = outputs[0]
        prediction = int(np.argmax(logits, axis=-1)[0])
        exp_logits = np.exp(logits - np.max(logits))
        confidence = float(np.max(exp_logits / np.sum(exp_logits)))
        return prediction, confidence
    except UnidentifiedImageError:
        raise ValueError("ไฟล์รูปภาพเสียหายหรือไม่ถูกต้อง")
    except Exception as e:
        raise Exception(f"Internal Inference Error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเท่านั้น (.jpg, .png)")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="ขนาดไฟล์ใหญ่เกินไป (จำกัด 5MB)")

    try:
        loop = asyncio.get_running_loop()
        prediction, confidence = await loop.run_in_executor(_get_executor(), predict_sync, content)
        return {"prediction_class": prediction, "confidence": confidence, "status": "success"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")


@app.get("/")
def root():
    return {
        "service": "insect-detector-api",
        "status": "running",
        "health_endpoint": "/health",
        "predict_endpoint": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
