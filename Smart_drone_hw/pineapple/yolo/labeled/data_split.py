# 分割訓練資料集
import os
import random
import shutil

labels=os.listdir('labels')
images=os.listdir('images')
print("Labels:", labels)
print("Images:", images)

os.makedirs('images/train', exist_ok=True)
os.makedirs('labels/train', exist_ok=True)
os.makedirs('images/val', exist_ok=True)
os.makedirs('labels/val', exist_ok=True)
os.makedirs('images/test', exist_ok=True)
os.makedirs('labels/test', exist_ok=True)

# 80%訓練集，10%驗證集，10%測試集
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 隨機打亂資料
random.seed(42)
random.shuffle(images)
random.shuffle(labels)

# 移訓練、驗證、測試集
for i, img in enumerate(images):
    if i < len(images) * train_ratio:
        shutil.move(os.path.join('images', img), os.path.join('images/train', img))
        shutil.move(os.path.join('labels', img.replace('.jpg', '.txt')), os.path.join('labels/train', img.replace('.jpg', '.txt')))
    elif i < len(images) * (train_ratio + val_ratio):
        shutil.move(os.path.join('images', img), os.path.join('images/val', img))
        shutil.move(os.path.join('labels', img.replace('.jpg', '.txt')), os.path.join('labels/val', img.replace('.jpg', '.txt')))
    else:
        shutil.move(os.path.join('images', img), os.path.join('images/test', img))
        shutil.move(os.path.join('labels', img.replace('.jpg', '.txt')), os.path.join('labels/test', img.replace('.jpg', '.txt')))

print("Data split completed.")  