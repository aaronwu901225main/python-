from ultralytics import YOLO
import os
import random
import shutil
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# -----------
# 2. 資料準備
# -----------
# 解壓資料集，假設叫 labeled_pineapple.zip
# !unzip labeled_pineapple.zip -d ./labeled_pineapple

random.seed(42)

use_val = False  # 設定: 要不要真的分出 validation set

all_images = list(Path('../labeled').glob('*.jpg'))
all_labels = list(Path('../labeled').glob('*.txt'))

# 過濾只保留有對應 label 的圖片
label_stems = set(p.stem for p in all_labels)
all_images = [img for img in all_images if img.stem in label_stems]

# 隨機打亂圖片順序
random.shuffle(all_images)

# 切成 train/test
train_size = int(0.8 * len(all_images))
train_images = all_images[:train_size]
test_images = all_images[train_size:]

# 建資料夾
os.makedirs(f'./dataset/images/train', exist_ok=True)
os.makedirs(f'./dataset/labels/train', exist_ok=True)

if use_val:
    os.makedirs(f'./dataset/images/val', exist_ok=True)
    os.makedirs(f'./dataset/labels/val', exist_ok=True)

# 複製 train
for img_path in train_images:
    label_path = Path('../labeled') / (img_path.stem + '.txt')
    shutil.copy(img_path, f'./dataset/images/train/{img_path.name}')
    shutil.copy(label_path, f'./dataset/labels/train/{label_path.name}')

# 複製 val 或 test
if use_val:
    for img_path in test_images:
        label_path = Path('../labeled') / (img_path.stem + '.txt')
        shutil.copy(img_path, f'./dataset/images/val/{img_path.name}')
        shutil.copy(label_path, f'./dataset/labels/val/{label_path.name}')
else:
    os.makedirs(f'./dataset/images/test', exist_ok=True)
    os.makedirs(f'./dataset/labels/test', exist_ok=True)
    for img_path in test_images:
        label_path = Path('../labeled') / (img_path.stem + '.txt')
        shutil.copy(img_path, f'./dataset/images/test/{img_path.name}')
        shutil.copy(label_path, f'./dataset/labels/test/{label_path.name}')

# -----------
# 3. 建立 data.yaml
# -----------
dataset_root = os.path.abspath('./dataset')
val_folder = 'images/val' if use_val else 'images/train'
with open('dataset/data.yaml', 'w') as f:
    f.write(f'''
path: {dataset_root}
train: images/train
val: {val_folder}
nc: 1
names: ['pineapple']
''')

# -----------
# 4. 訓練模型 + 儲存
# -----------
model = YOLO('yolov8n.pt')
model.train(data='dataset/data.yaml', epochs=100, imgsz=640)

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

image_paths = list(Path('./dataset/images/test').glob('*.jpg'))

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
    label_file = Path('./dataset/labels/test') / (img_path.stem + '.txt')
    if label_file.exists():
        gt_count = sum(1 for _ in open(label_file))
    else:
        print(f"警告：找不到標籤 {label_file}，將GT數量設為0！")
        gt_count = 0

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

# -----------
# 7. 畫出散點圖 (預測數量 vs 真實數量)
# -----------
plt.figure(figsize=(8,6))
plt.scatter(gt_counts, predict_counts, color='blue', label='Predictions')
plt.plot([0, max(gt_counts)], [0, max(gt_counts)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Ground Truth Count')
plt.ylabel('Predicted Count')
plt.title('Predicted vs Ground Truth Count')
plt.legend()
plt.grid(True)
plt.savefig('./results/pred_vs_gt_plot.png')
plt.close()