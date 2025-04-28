### train_and_infer_pineapple.py

# -----------
# 1. 環境準備
# -----------
from ultralytics import YOLO
import os
import random
import shutil
import pandas as pd
import cv2
from pathlib import Path

# -----------
# 2. 資料準備
# -----------
# 解壓資料集，假設叫 labeled_pineapple.zip
# !unzip labeled_pineapple.zip -d ./labeled_pineapple

random.seed(42)

all_images = list(Path('../dataset/images').glob('*.jpg'))
all_labels = list(Path('../dataset/labels').glob('*.txt'))

# 隨機打散切分 (8:2)
combined = list(zip(all_images, all_labels))
random.shuffle(combined)
train_size = int(0.8 * len(combined))
train_data = combined[:train_size]
test_data = combined[train_size:]

# 建資料夾
for split in ['train', 'val']:
    os.makedirs(f'./dataset/images/{split}', exist_ok=True)
    os.makedirs(f'./dataset/labels/{split}', exist_ok=True)

# 移動檔案到新資料夾
for img_path, label_path in train_data:
    shutil.copy(img_path, f'./dataset/images/train/{img_path.name}')
    shutil.copy(label_path, f'./dataset/labels/train/{label_path.name}')

for img_path, label_path in test_data:
    shutil.copy(img_path, f'./dataset/images/val/{img_path.name}')
    shutil.copy(label_path, f'./dataset/labels/val/{label_path.name}')

# -----------
# 3. 建立 data.yaml
# -----------
with open('dataset/data.yaml', 'w') as f:
    f.write('''
path: ./dataset
train: images/train
val: images/val
nc: 1
names: ['pineapple']
''')

# -----------
# 4. 訓練模型 + 儲存
# -----------
model = YOLO('yolov8n.pt')
model.train(data='dataset/data.yaml', epochs=50, imgsz=640)

# 儲存訓練好的模型
os.makedirs('./saved_models', exist_ok=True)
save_path = './saved_models/pineapple_detector.pt'
model.save(save_path)

# -----------
# 5. 載入模型 & 推論測試集 & 畫出來
# -----------
output_dir = './predicted_images'
os.makedirs(output_dir, exist_ok=True)

# 載入剛剛訓練好的模型
model = YOLO(save_path)

image_paths = list(Path('./dataset/images/val').glob('*.jpg'))

predict_counts = []
gt_counts = []
image_names = []

for img_path in image_paths:
    results = model(img_path)  # 推論
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 畫框
    img = cv2.imread(str(img_path))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(f'{output_dir}/{img_path.name}', img)

    # 計數
    pred_count = len(boxes)
    label_file = Path('./dataset/labels/val') / (img_path.stem + '.txt')
    gt_count = sum(1 for _ in open(label_file))

    predict_counts.append(pred_count)
    gt_counts.append(gt_count)
    image_names.append(img_path.name)

# -----------
# 6. 輸出預測數量 vs GT 數量表格
# -----------
df = pd.DataFrame({
    'Image': image_names,
    'Predicted_Count': predict_counts,
    'Ground_Truth_Count': gt_counts
})

os.makedirs('./results', exist_ok=True)
df.to_csv('./results/pred_vs_gt.csv', index=False)
print("✅ 預測數量 vs GT 數量表格已儲存為: ./results/pred_vs_gt.csv")