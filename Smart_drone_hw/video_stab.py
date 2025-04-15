import cv2
import numpy as np

# 讀取影片
cap = cv2.VideoCapture('DJI_0025_W.MP4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 取得影片的第一幀作為背景
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video.")
    exit()

# 影片的幾何尺寸
frame_height, frame_width = frame.shape[:2]

# 用來儲存穩定後的影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('stabilized_video_with_moving_objects.mp4', fourcc, 30.0, (frame_width, frame_height))

# 影像穩定化: 使用光流法來消除鏡頭晃動
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
transforms = []

n=0
# 讀取並處理每一幀
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    print("Processing frame", n)
    n+=1
    
    # 轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 計算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 從光流中計算平移和旋轉
    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    # 儲存變換
    transforms.append((dx, dy))

    # 更新上一幀為當前幀
    prev_gray = gray

# 重置影片指針
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 進行穩定化並生成最終影片
for dx, dy in transforms:
    print("正在處理平移: dx={}, dy={}".format(dx, dy))
    ret, frame = cap.read()
    if not ret:
        break

    # 應用光流變換來穩定化影像
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    stabilized_frame = cv2.warpAffine(frame, matrix, (frame_width, frame_height))

    # 使用第一幀作為背景，並保持移動物體
    fgmask = cv2.absdiff(stabilized_frame, frame)
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    _, fgmask = cv2.threshold(fgmask, 30, 255, cv2.THRESH_BINARY)

    # 將移動物體與背景分離
    stabilized_frame[fgmask == 0] = frame[fgmask == 0]  # 保持背景固定
    stabilized_frame[fgmask != 0] = frame[fgmask != 0]  # 保持移動物體不變

    # 寫入穩定化後的影片
    out.write(stabilized_frame)

# 完成後釋放資源
cap.release()
out.release()

print("Video stabilized with fixed background and moving objects. Saved as 'stabilized_video_with_moving_objects.mp4'.")
