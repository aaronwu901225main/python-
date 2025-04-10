import cv2
import numpy as np

def gray_world_balance(img):
    # 轉換為RGB格式並轉為浮點型以便計算
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 計算各通道的平均值
    avg_r = np.mean(img_rgb[:, :, 0])
    avg_g = np.mean(img_rgb[:, :, 1])
    avg_b = np.mean(img_rgb[:, :, 2])
    avg = (avg_r + avg_g + avg_b) / 3.0  # 目標平均值
    
    # 計算增益係數
    gain_r = avg / avg_r if avg_r != 0 else 1.0
    gain_g = avg / avg_g if avg_g != 0 else 1.0
    gain_b = avg / avg_b if avg_b != 0 else 1.0
    
    # 應用增益並限制數值範圍
    img_rgb[:, :, 0] = np.clip(img_rgb[:, :, 0] * gain_r, 0, 255)
    img_rgb[:, :, 1] = np.clip(img_rgb[:, :, 1] * gain_g, 0, 255)
    img_rgb[:, :, 2] = np.clip(img_rgb[:, :, 2] * gain_b, 0, 255)
    
    # 轉換回BGR格式並返回
    balanced_img = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return balanced_img

# 讀取圖像並處理
input_img = cv2.imread('white-balance-auto-sample-image_1465-7.jpg')
output_img = gray_world_balance(input_img)

# 保存結果
cv2.imwrite('output.jpg', output_img)