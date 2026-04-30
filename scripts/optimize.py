import csv
import json
import logging
import os
import time

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from onnxruntime.quantization import QuantType, quantize_dynamic


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_MODEL_DIR = os.path.join(BASE_DIR, "model", "fine_tuned_insect_v2")
MODEL_DIR = os.path.join(BASE_DIR, "model", "onnx_v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
WORK_DIR = os.path.join(BASE_DIR, "model", "optimization")

ORIGINAL_MODEL_PATH = os.path.join(ORIGINAL_MODEL_DIR, "model.safetensors")
ORIGINAL_PREPROCESSOR_PATH = os.path.join(ORIGINAL_MODEL_DIR, "preprocessor_config.json")
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_config.json")
QUANTIZED_MODEL_PATH = os.path.join(WORK_DIR, "model_quantized.onnx")
METRICS_CSV = os.path.join(REPORTS_DIR, "optimization_metrics.csv")


def ensure_paths() -> None:
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลต้นฉบับ: {ORIGINAL_MODEL_PATH}")
    if not os.path.exists(ORIGINAL_PREPROCESSOR_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์ preprocessor ต้นฉบับ: {ORIGINAL_PREPROCESSOR_PATH}")
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {BASE_MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์ preprocessor: {PREPROCESSOR_PATH}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)


def load_preprocessor_config() -> dict:
    with open(PREPROCESSOR_PATH, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_original_preprocessor_config() -> dict:
    with open(ORIGINAL_PREPROCESSOR_PATH, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def extract_target_size(preprocessor_cfg: dict) -> tuple[int, int]:
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


def build_dummy_input(preprocessor_cfg: dict) -> np.ndarray:
    width, height = extract_target_size(preprocessor_cfg)

    # ใช้ข้อมูลสุ่มเพื่อให้ benchmark ตัว inference ได้จริงโดยไม่ต้องพึ่งรูปภายนอก
    dummy = np.random.rand(1, 3, height, width).astype(np.float32)

    if preprocessor_cfg.get("do_normalize", True):
        image_mean = np.array(preprocessor_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        image_std = np.array(preprocessor_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        dummy = (dummy - image_mean.reshape(1, 3, 1, 1)) / image_std.reshape(1, 3, 1, 1)

    return dummy


def build_original_input(preprocessor_cfg: dict) -> torch.Tensor:
    width, height = extract_target_size(preprocessor_cfg)
    image = Image.fromarray((np.random.rand(height, width, 3) * 255).astype(np.uint8), mode="RGB")
    processor = AutoImageProcessor.from_pretrained(ORIGINAL_MODEL_DIR, local_files_only=True, use_fast=False)
    return processor(images=image, return_tensors="pt").pixel_values


def create_original_model() -> MobileNetV2ForImageClassification:
    model = MobileNetV2ForImageClassification.from_pretrained(
        ORIGINAL_MODEL_DIR,
        local_files_only=True,
    )
    model.eval()
    return model


def measure_original_latency(model: MobileNetV2ForImageClassification, pixel_values: torch.Tensor, iterations: int = 50) -> float:
    with torch.no_grad():
        _ = model(pixel_values)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(pixel_values)
    end_time = time.perf_counter()

    return ((end_time - start_time) / iterations) * 1000


def create_session(model_path: str) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])


def measure_latency(session: ort.InferenceSession, dummy_input: np.ndarray, iterations: int = 50) -> float:
    input_name = session.get_inputs()[0].name

    # Warm up ก่อนวัดจริง
    _ = session.run(None, {input_name: dummy_input})

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()

    return ((end_time - start_time) / iterations) * 1000


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    ensure_paths()
    torch.set_num_threads(1)

    original_preprocessor_cfg = load_original_preprocessor_config()
    original_input = build_original_input(original_preprocessor_cfg)

    print("[1/3] กำลังวัดผลโมเดลต้นฉบับ (Original)...")
    original_model = create_original_model()
    original_latency = measure_original_latency(original_model, original_input)
    original_size = file_size_mb(ORIGINAL_MODEL_PATH)

    preprocessor_cfg = load_preprocessor_config()
    dummy_input = build_dummy_input(preprocessor_cfg)

    print("[2/3] กำลังวัดผลโมเดล ONNX ต้นฉบับ...")
    base_session = create_session(BASE_MODEL_PATH)
    base_latency = measure_latency(base_session, dummy_input)
    base_size = file_size_mb(BASE_MODEL_PATH)

    print("[3/3] กำลังสร้างโมเดลที่ถูก quantize และวัดผล...")
    previous_logging_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    quantize_dynamic(
        model_input=BASE_MODEL_PATH,
        model_output=QUANTIZED_MODEL_PATH,
        weight_type=QuantType.QInt8,
    )
    logging.getLogger().setLevel(previous_logging_level)

    quantized_session = create_session(QUANTIZED_MODEL_PATH)
    quantized_latency = measure_latency(quantized_session, dummy_input)
    quantized_size = file_size_mb(QUANTIZED_MODEL_PATH)

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["model_type", "latency_ms", "size_mb"])
        writer.writerow(["Original", round(original_latency, 4), round(original_size, 4)])
        writer.writerow(["ONNX Base", round(base_latency, 4), round(base_size, 4)])
        writer.writerow(["Quantized ONNX", round(quantized_latency, 4), round(quantized_size, 4)])

    print("\n" + "=" * 50)
    print("สรุปผลการ Optimization")
    print("=" * 50)
    print(f"{'Model Type':<18} | {'Latency (ms)':<12} | {'Size (MB)':<10}")
    print("-" * 50)
    print(f"{'Original':<18} | {original_latency:>12.2f} | {original_size:>10.2f}")
    print(f"{'ONNX Base':<18} | {base_latency:>12.2f} | {base_size:>10.2f}")
    print(f"{'Quantized ONNX':<18} | {quantized_latency:>12.2f} | {quantized_size:>10.2f}")
    print("=" * 50)
    print(f"บันทึกผลไว้ที่: {METRICS_CSV}")
    print(f"ไฟล์ quantized เก็บไว้ที่: {QUANTIZED_MODEL_PATH}")


if __name__ == "__main__":
    main()