import torch
import os
from PIL import Image
from datasets import Dataset, Features, ClassLabel, Image as ImageFeature
from transformers import (
    AutoImageProcessor, 
    MobileNetV2ForImageClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torchvision import transforms

# 1. ตั้งค่า Path หลัก
# ใช้ os.getcwd() เพื่อให้แน่ใจว่า path เริ่มต้นจากจุดที่รันสคริปต์
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ip102")

# 2. ฟังก์ชันโหลด Data ที่รองรับโครงสร้างโฟลเดอร์แบบแบ่ง Split และ Class ID
def load_ip102_data(txt_file, split_folder):
    image_paths, labels = [], []
    full_txt_path = os.path.join(DATA_DIR, txt_file)
    # โครงสร้าง: data/ip102/classification/[train/val/test]/[class_id]/[filename]
    base_split_dir = os.path.join(DATA_DIR, "classification", split_folder)
    
    if not os.path.exists(full_txt_path):
        print(f"⚠️ ไม่พบไฟล์ดัชนี: {full_txt_path}")
        return {"image": [], "label": []}

    with open(full_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_name = parts[0]
                class_id = parts[1]
                img_path = os.path.normpath(os.path.join(base_split_dir, class_id, file_name))
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(int(class_id))
    
    print(f"✅ โหลดสำเร็จ ({split_folder}): {len(image_paths)} ภาพ")
    return {"image": image_paths, "label": labels}

# 3. เตรียม Dataset และ Classes
print("📦 กำลังจัดเตรียมข้อมูล...")
train_data = load_ip102_data("train.txt", "train")
val_data = load_ip102_data("val.txt", "val")
test_data = load_ip102_data("test.txt", "test")

with open(os.path.join(DATA_DIR, "classes.txt"), "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# กำหนด Features ให้ Dataset รู้จักประเภทข้อมูล
features = Features({
    "image": ImageFeature(),
    "label": ClassLabel(names=class_names)
})

train_ds = Dataset.from_dict(train_data, features=features)
val_ds = Dataset.from_dict(val_data, features=features)
test_ds = Dataset.from_dict(test_data, features=features)

# 4. Data Augmentation (ช่วยเพิ่ม Accuracy โดยการสร้างความหลากหลายให้รูป)
train_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

val_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")

# แก้ไขฟังก์ชัน Transform ให้คืนค่า pixel_values และ labels ให้ถูกต้อง
def train_transforms(examples):
    examples["pixel_values"] = [
        processor(train_aug(img.convert("RGB")), return_tensors="pt")["pixel_values"][0] 
        for img in examples["image"]
    ]
    examples["labels"] = examples["label"]
    return examples

def val_transforms(examples):
    examples["pixel_values"] = [
        processor(val_aug(img.convert("RGB")), return_tensors="pt")["pixel_values"][0] 
        for img in examples["image"]
    ]
    examples["labels"] = examples["label"]
    return examples


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "labels": torch.tensor([example["labels"] for example in examples], dtype=torch.long),
    }

# นำการแปลงข้อมูลไปผูกกับ Dataset (จะทำงานแบบ Dynamic ขณะเทรน)
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

# 5. โหลดโมเดล MobileNetV2 พร้อมตั้งค่า Label Mapping
model = MobileNetV2ForImageClassification.from_pretrained(
    "google/mobilenet_v2_1.0_224",
    num_labels=len(class_names),
    id2label={i: l for i, l in enumerate(class_names)},
    label2id={l: i for i, l in enumerate(class_names)},
    ignore_mismatched_sizes=True
)

# 6. ตั้งค่าการเทรน (Optimized for RTX 3050 6GB)
training_args = TrainingArguments(
    output_dir="../model/checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,              # LR ที่ต่ำลงช่วยให้การจูนละเอียดขึ้น
    per_device_train_batch_size=32,   # เหมาะสมกับ VRAM 6GB
    num_train_epochs=30,             # เพิ่มจำนวนรอบเพื่อให้ Accuracy สูงขึ้น
    weight_decay=0.01,
    lr_scheduler_type="cosine",      # ลด LR แบบโค้งในช่วงท้ายเพื่อความนิ่ง
    warmup_ratio=0.1,
    fp16=True,                       # ใช้พลัง Tensor Core ของ RTX 3050
    load_best_model_at_end=True,     # ดึงโมเดลที่แม่นที่สุดมาเซฟตอนจบ
    metric_for_best_model="accuracy",
    logging_steps=50,
    
    # จุดสำคัญ: ตั้งเป็น False เพื่อป้องกัน KeyError: 'image'
    remove_unused_columns=False, 
    
    save_total_limit=3               # เก็บแค่ 3 checkpoint ล่าสุด ประหยัดพื้นที่
)

# 7. ฟังก์ชันวัดผล
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": (predictions == torch.tensor(labels)).float().mean().item()}

# 8. เริ่มต้น Trainer พร้อมระบบ Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # หยุดถ้าไม่ดีขึ้นติดต่อกัน 5 รอบ
)

print("🚀 เริ่มการเทรนเวอร์ชัน High Accuracy บน GPU...")
trainer.train()

# 9. วัดผลกับชุดข้อมูล Test (คะแนนสอบจริง)
print("🧪 กำลังประเมินผลกับชุดข้อมูล Test...")
test_results = trainer.evaluate(test_ds)
print(f"🎯 Final Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")

# 10. บันทึกโมเดลฉบับปรับปรุง
output_path = "../model/fine_tuned_insect_v2"
model.save_pretrained(output_path)
processor.save_pretrained(output_path)
print(f"✅ สำเร็จ! โมเดล v2 ถูกบันทึกไว้ที่: {output_path}")