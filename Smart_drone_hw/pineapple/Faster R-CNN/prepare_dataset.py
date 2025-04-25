import os
import shutil
import random

def prepare_dataset(source_dir='labeled', output_dir='project', train_ratio=0.8):
    # 建立資料夾
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 取得所有圖片檔
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    images.sort()

    # 打亂
    random.shuffle(images)

    # 切分
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # 複製資料
    for split_name, split_images in [('train', train_images), ('test', test_images)]:
        for img_name in split_images:
            label_name = img_name.replace('.jpg', '.txt')
            shutil.copy(os.path.join(source_dir, img_name), os.path.join(output_dir, 'images', split_name, img_name))
            shutil.copy(os.path.join(source_dir, label_name), os.path.join(output_dir, 'labels', split_name, label_name))

    print(f"切分完成！訓練集：{len(train_images)} 張，測試集：{len(test_images)} 張。")

if __name__ == "__main__":
    print(f"當前工作目錄：{os.getcwd()}")  # 確認當前工作目錄
    prepare_dataset()