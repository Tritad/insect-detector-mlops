import torch
import time
import os
import csv
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from optimum.onnxruntime import ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 1. ตั้งค่าพื้นฐาน
model_id = "google/mobilenet_v2_1.0_224"
model_path = "model"
os.makedirs(model_path, exist_ok=True)

def measure_latency(model, processor, iterations=50):
    """ฟังก์ชันวัดความเร็วเฉลี่ย (Latency) ต่อ 1 รูป"""
    # สร้างข้อมูลภาพจำลอง (Dummy data)
    pixel_values = torch.randn(1, 3, 224, 224)
    
    # Warm up (รันครั้งแรกเพื่อให้ระบบพร้อม)
    with torch.no_grad():
        _ = model(pixel_values)
    
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(pixel_values)
    end_time = time.time()
    
    return ((end_time - start_time) / iterations) * 1000 # คืนค่าเป็น ms

# --- STEP 1: ORIGINAL MODEL (BASELINE) ---
print("\n[1/3] กำลังวัดผลโมเดลต้นฉบับ (Original)...")
original_model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)

orig_latency = measure_latency(original_model, processor)
# คำนวณขนาดไฟล์โดยประมาณ (MB)
orig_size = sum(p.numel() for p in original_model.parameters()) * 4 / (1024 * 1024)

# --- STEP 2: CONVERT TO ONNX ---
print("[2/3] กำลังแปลงโมเดลเป็นรูปแบบ ONNX...")
onnx_model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
onnx_model.save_pretrained(f"{model_path}/onnx")
# หมายเหตุ: การวัดผล ONNX ในสคริปต์นี้จะใช้โมเดลที่โหลดมาใหม่
onnx_latency = measure_latency(onnx_model, processor)

# --- STEP 3: DYNAMIC QUANTIZATION ---
print("[3/3] กำลังทำ Dynamic Quantization (ลดขนาดโมเดล)...")
quantizer = ORTQuantizer.from_pretrained(onnx_model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=f"{model_path}/quantized", quantization_config=dqconfig)

# โหลดโมเดลที่ Quantize แล้วมาวัดผล
q_model = ORTModelForImageClassification.from_pretrained(f"{model_path}/quantized")
q_latency = measure_latency(q_model, processor)

# ตรวจสอบขนาดไฟล์จริงใน Disk
def get_size(path):
    size = os.path.getsize(path) / (1024 * 1024)
    return size

onnx_file_size = get_size(f"{model_path}/onnx/model.onnx")
q_file_size = get_size(f"{model_path}/quantized/model_quantized.onnx")

metrics_dir = "reports"
os.makedirs(metrics_dir, exist_ok=True)
metrics_csv = os.path.join(metrics_dir, "optimization_metrics.csv")

rows = [
    ["Original", round(orig_latency, 4), round(orig_size, 4)],
    ["ONNX", round(onnx_latency, 4), round(onnx_file_size, 4)],
    ["Quantized", round(q_latency, 4), round(q_file_size, 4)],
]

with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model_type", "latency_ms", "size_mb"])
    writer.writerows(rows)

# --- สรุปผลลัพธ์ ---
print("\n" + "="*45)
print("📊 ตารางเปรียบเทียบผลการ Optimization")
print("="*45)
print(f"{'Model Type':<15} | {'Latency (ms)':<12} | {'Size (MB)':<10}")
print("-" * 45)
print(f"{'Original':<15} | {orig_latency:>12.2f} | {orig_size:>10.2f}")
print(f"{'ONNX':<15} | {onnx_latency:>12.2f} | {onnx_file_size:>10.2f}")
print(f"{'Quantized':<15} | {q_latency:>12.2f} | {q_file_size:>10.2f}")
print("="*45)
print(f"✅ แปลงโมเดลเสร็จสิ้น! ไฟล์เก็บอยู่ที่โฟลเดอร์ model/")
print(f"📝 บันทึกตารางผลไว้ที่: {metrics_csv}")