# สร้างไฟล์ export_v2.py แล้วรัน
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor

model_id = "../model/fine_tuned_insect_v2"
save_path = "../model/onnx_v2"

model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
processor = AutoImageProcessor.from_pretrained(model_id)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("🚀 แปลงเป็น ONNX v2 เรียบร้อย!")