"""
Script để phân tích và chọn ngưỡng tin cậy (confidence threshold) tốt nhất cho API
Chạy: python scripts/analyze_confidence_threshold.py
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ===== CẤU HÌNH =====
DATASET_PATH = r"E:\Download\Dataset"  # THAY ĐỔI ĐƯỜNG DẪN NÀY
MODEL_PATH = r"BEST_MODEL.pth"
CLASS_NAMES_PATH = r"class_names.json"
OUTPUT_DIR = r"threshold_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== LOAD MODEL =====
print("Đang load model...")
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

num_classes = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)

# Load state dict và lọc bỏ các key total_ops/total_params từ THOP profiling
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
filtered_state = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
model.load_state_dict(filtered_state, strict=False)

model.to(device)
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===== ĐỌC DATASET =====
print("Đang đọc dataset...")

# Kiểm tra đường dẫn dataset
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ LỖI: Không tìm thấy dataset tại: {DATASET_PATH}")
    print(f"\n📝 HƯỚNG DẪN:")
    print(f"   1. Mở file: scripts/analyze_confidence_threshold.py")
    print(f"   2. Sửa dòng 20: DATASET_PATH = r\"ĐƯỜNG_DẪN_DATASET_CỦA_BẠN\"")
    print(f"   3. Ví dụ: DATASET_PATH = r\"E:\\Download\\Dataset\"")
    exit(1)

image_paths, labels = [], []

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_dir): 
        continue
    for file in os.listdir(class_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(class_dir, file))
            labels.append(class_name)

if len(image_paths) == 0:
    print(f"\n❌ LỖI: Không tìm thấy ảnh nào trong dataset!")
    print(f"   Đường dẫn: {DATASET_PATH}")
    print(f"   Cấu trúc mong đợi:")
    print(f"   {DATASET_PATH}/")
    print(f"   ├── Tomato___Late_blight/")
    print(f"   │   ├── image1.jpg")
    print(f"   │   └── image2.jpg")
    print(f"   └── Pepper___Bacterial_spot/")
    print(f"       └── image3.jpg")
    exit(1)

print(f"✓ Tìm thấy {len(image_paths)} ảnh từ {len(set(labels))} lớp")

class_to_idx = {c: i for i, c in enumerate(class_names)}
label_indices = [class_to_idx[l] for l in labels]

# Chia tập test (10% cuối cùng)
_, X_test, _, y_test = train_test_split(
    image_paths, label_indices, test_size=0.1, stratify=label_indices, random_state=42
)

print(f"Số ảnh test: {len(X_test)}")

# ===== DỰ ĐOÁN VÀ LƯU XÁC SUẤT =====
print("\nĐang dự đoán trên tập test...")
all_confidences = []  # Xác suất cao nhất của mỗi ảnh
correct_confidences = []  # Xác suất khi dự đoán đúng
wrong_confidences = []  # Xác suất khi dự đoán sai

with torch.no_grad():
    for img_path, true_label in tqdm(zip(X_test, y_test), total=len(X_test)):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            max_prob, pred_label = probs.max(1)
            
            confidence = max_prob.item()
            all_confidences.append(confidence)
            
            if pred_label.item() == true_label:
                correct_confidences.append(confidence)
            else:
                wrong_confidences.append(confidence)
        except Exception as e:
            print(f"Lỗi xử lý {img_path}: {e}")
            continue

all_confidences = np.array(all_confidences)
correct_confidences = np.array(correct_confidences)
wrong_confidences = np.array(wrong_confidences)

# ===== PHÂN TÍCH NGƯỠNG =====
print("\n" + "="*60)
print("PHÂN TÍCH CONFIDENCE THRESHOLD")
print("="*60)

# Thống kê cơ bản
print(f"\nTổng số ảnh test: {len(all_confidences)}")
print(f"Số ảnh dự đoán đúng: {len(correct_confidences)} ({len(correct_confidences)/len(all_confidences)*100:.2f}%)")
print(f"Số ảnh dự đoán sai: {len(wrong_confidences)} ({len(wrong_confidences)/len(all_confidences)*100:.2f}%)")

print(f"\nConfidence trung bình:")
print(f"  - Toàn bộ: {all_confidences.mean():.4f}")
print(f"  - Dự đoán đúng: {correct_confidences.mean():.4f}")
print(f"  - Dự đoán sai: {wrong_confidences.mean():.4f}")

# Phân tích theo các ngưỡng khác nhau
print("\n" + "-"*60)
print("ĐÁNH GIÁ CÁC NGƯỠNG KHÁC NHAU")
print("-"*60)
print(f"{'Threshold':<12} {'Accept%':<10} {'Accuracy':<12} {'Reject%':<10}")
print("-"*60)

threshold_results = []
for threshold in np.arange(0.5, 1.0, 0.05):
    accepted_mask = all_confidences >= threshold
    num_accepted = accepted_mask.sum()
    
    if num_accepted > 0:
        accepted_correct = np.sum(np.array(correct_confidences) >= threshold)
        accuracy_at_threshold = accepted_correct / num_accepted
    else:
        accuracy_at_threshold = 0
    
    accept_rate = num_accepted / len(all_confidences)
    reject_rate = 1 - accept_rate
    
    threshold_results.append({
        'threshold': threshold,
        'accept_rate': accept_rate,
        'accuracy': accuracy_at_threshold,
        'reject_rate': reject_rate
    })
    
    print(f"{threshold:.2f}         {accept_rate*100:6.2f}%    {accuracy_at_threshold*100:6.2f}%      {reject_rate*100:6.2f}%")

# ===== VISUALIZATION =====
print("\nĐang tạo biểu đồ...")

# 1. Histogram của confidence scores
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Distribution của correct vs wrong
axes[0, 0].hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0, 0].hist(wrong_confidences, bins=50, alpha=0.7, label='Wrong', color='red', edgecolor='black')
axes[0, 0].set_xlabel('Confidence Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Phân bố Confidence: Đúng vs Sai')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Cumulative distribution
axes[0, 1].hist(all_confidences, bins=50, cumulative=True, alpha=0.7, color='blue', edgecolor='black')
axes[0, 1].set_xlabel('Confidence Score')
axes[0, 1].set_ylabel('Cumulative Count')
axes[0, 1].set_title('Phân bố tích lũy Confidence')
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Accept Rate vs Threshold
thresholds = [r['threshold'] for r in threshold_results]
accept_rates = [r['accept_rate'] * 100 for r in threshold_results]
axes[1, 0].plot(thresholds, accept_rates, marker='o', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('Accept Rate (%)')
axes[1, 0].set_title('Tỷ lệ Accept theo Threshold')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Accuracy at different thresholds
accuracies = [r['accuracy'] * 100 for r in threshold_results]
axes[1, 1].plot(thresholds, accuracies, marker='s', linewidth=2, markersize=6, color='green')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Accuracy khi áp dụng Threshold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ: {os.path.join(OUTPUT_DIR, 'confidence_analysis.png')}")

# ===== KHUYẾN NGHỊ =====
print("\n" + "="*60)
print("KHUYẾN NGHỊ NGƯỠNG TIN CẬY")
print("="*60)

# Tìm ngưỡng tối ưu (balance giữa accuracy và accept rate)
# Ngưỡng tốt: accuracy > 95% và accept rate > 80%
optimal_thresholds = [
    r for r in threshold_results 
    if r['accuracy'] >= 0.95 and r['accept_rate'] >= 0.80
]

if optimal_thresholds:
    best = optimal_thresholds[0]
    print(f"\n✅ NGƯỠNG ĐỀ XUẤT: {best['threshold']:.2f}")
    print(f"   - Accept Rate: {best['accept_rate']*100:.2f}%")
    print(f"   - Accuracy: {best['accuracy']*100:.2f}%")
    print(f"   - Reject Rate: {best['reject_rate']*100:.2f}%")
else:
    # Tìm threshold có accuracy cao nhất mà vẫn accept > 70%
    viable = [r for r in threshold_results if r['accept_rate'] >= 0.70]
    if viable:
        best = max(viable, key=lambda x: x['accuracy'])
        print(f"\n⚠️  NGƯỠNG ĐỀ XUẤT (relaxed): {best['threshold']:.2f}")
        print(f"   - Accept Rate: {best['accept_rate']*100:.2f}%")
        print(f"   - Accuracy: {best['accuracy']*100:.2f}%")
        print(f"   - Reject Rate: {best['reject_rate']*100:.2f}%")
    else:
        print("\n⚠️  Không tìm thấy ngưỡng phù hợp. Xem xét lại mô hình.")

# Lưu kết quả
results_file = os.path.join(OUTPUT_DIR, 'threshold_analysis.json')
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'statistics': {
            'total_samples': len(all_confidences),
            'correct_predictions': len(correct_confidences),
            'wrong_predictions': len(wrong_confidences),
            'avg_confidence_all': float(all_confidences.mean()),
            'avg_confidence_correct': float(correct_confidences.mean()),
            'avg_confidence_wrong': float(wrong_confidences.mean())
        },
        'threshold_analysis': threshold_results
    }, f, indent=2)

print(f"\n✅ Kết quả chi tiết đã lưu tại: {results_file}")
print("\n" + "="*60)
