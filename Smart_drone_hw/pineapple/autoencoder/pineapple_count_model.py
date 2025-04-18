# %% [markdown]
# # 讀取資料

# %%
import os
import random
import numpy as np
import cv2

# 讀取所有圖片檔案和標註檔案
def load_data_from_directory(directory_path):
    images = []
    labels = []
    
    # 讀取資料夾中的所有圖片和標註檔案
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory_path, filename)
            label_path = os.path.join(directory_path, filename.replace(".jpg", ".txt"))
            
            # 讀取圖片
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換為RGB格式
            
            # 讀取標註
            with open(label_path, 'r') as label_file:
                coords = []
                for line in label_file.readlines():
                    parts = line.split()
                    x_min, y_min, width, height = map(float, parts[1:])
                    coords.append([x_min, y_min, width, height])
                coords = np.array(coords)
            
            # 儲存圖片和標註
            images.append(image)
            labels.append(coords)
    
    return images, labels

# 讀取資料
directory_path = r'C:\Users\AaronWu\Documents\GitHub\python-\Smart_drone_hw\pineapple\autoencoder\labeled'  # 替換為實際的資料夾路徑
images, labels = load_data_from_directory(directory_path)


# %% [markdown]
# # 切分訓練資料與測試資料

# %%
from sklearn.model_selection import train_test_split

# 將資料分為訓練資料和測試資料
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"訓練資料數量：{len(X_train)}")
print(f"測試資料數量：{len(X_test)}")


# %% [markdown]
# # 處理資料

# %%
from tensorflow.keras.preprocessing.image import img_to_array

# 圖片預處理函數，將圖片大小調整為 4096x2160
def preprocess_image(image, target_size=(4096, 2160)):
    image = cv2.resize(image, target_size)  # 調整圖片大小
    image = img_to_array(image) / 255.0  # 標準化
    return image

# 預處理訓練資料和測試資料
X_train = np.array([preprocess_image(img) for img in X_train])
X_test = np.array([preprocess_image(img) for img in X_test])


# %% [markdown]
# # 構建自編碼器模型

# %%
import tensorflow as tf
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape=(4096, 2160, 3)):
    input_img = layers.Input(shape=input_shape)
    # 編碼器部分
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # 解碼器部分
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# 輸入形狀設置為 4096x2160
input_shape = (4096, 2160, 3)
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

# %% [markdown]
# # 訓練模型

# %%
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# 自定義回呼函數，來實現進度條
class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epochs = self.params['epochs']
        self.epochs_progress = tqdm(total=self.epochs, desc="Epochs", position=0)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_progress.update(1)

    def on_batch_end(self, batch, logs=None):
        self.epochs_progress.set_postfix(loss=logs['loss'], epoch=self.params['epochs'])

# 使用自定義的回呼函數
progress_bar = TQDMProgressBar()

# 訓練模型並添加進度條
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test), callbacks=[progress_bar])


# %% [markdown]
# # 預測與保存結果

# %%
# 進行預測
predicted_images = autoencoder.predict(X_test)

# 假設實際的鳳梨數量（Ground Truth）
gt_pineapples = [len(label) for label in y_test]

# 預測的鳳梨數量（可以根據模型輸出進行簡單估算）
predicted_pineapples = [np.sum(pred > 0.5) for pred in predicted_images]  # 假設高於0.5的區域為鳳梨

# 比較預測結果與真實結果
print("Ground Truth vs Prediction:")
for i in range(len(gt_pineapples)):
    print(f"圖片 {i + 1}: 實際鳳梨數量={gt_pineapples[i]}, 預測鳳梨數量={predicted_pineapples[i]}")


# %% [markdown]
# # 儲存結果

# %%
import csv

# 儲存預測和 GT 的對比結果
with open('predicted_vs_gt.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["圖片編號", "Ground Truth", "Prediction"])
    for i in range(len(gt_pineapples)):
        writer.writerow([i + 1, gt_pineapples[i], predicted_pineapples[i]])



