import cv2
import numpy as np

# 儲存滑鼠點選的 4 個點
points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")

# 載入背景圖片和要貼上去的圖片
background = cv2.imread('background.png')  # 要選範圍的底圖
overlay = cv2.imread('overlay.jpg')        # 要貼上去的圖片

# 顯示背景圖片，讓使用者點 4 個點
cv2.namedWindow("Select 4 points")
cv2.setMouseCallback("Select 4 points", mouse_callback)

while True:
    temp = background.copy()
    for p in points:
        cv2.circle(temp, p, 5, (0, 255, 0), -1)
    cv2.imshow("Select 4 points", temp)
    key = cv2.waitKey(1)
    if key == 27 or len(points) == 4:  # 按 Esc 或點滿 4 點就繼續
        break

cv2.destroyAllWindows()

if len(points) != 4:
    print("請選滿 4 個點")
    exit()

# 將 overlay 圖片轉換成與點對應的尺寸
h, w = overlay.shape[:2]

# 定義 overlay 的四個角落 (原始座標)
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

# 使用你選的 4 個點當作目標座標
dst_pts = np.float32(points)

# 計算透視變換矩陣
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 套用透視變換到 overlay 圖片上
warped_overlay = cv2.warpPerspective(overlay, matrix, (background.shape[1], background.shape[0]))

# 建立遮罩
mask = np.zeros_like(background, dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(dst_pts), (255, 255, 255))

# 使用遮罩將原背景清掉貼圖區域，然後加上 warped_overlay
masked_bg = cv2.bitwise_and(background, cv2.bitwise_not(mask))
result = cv2.add(masked_bg, warped_overlay)

# 顯示結果
cv2.imshow("Result", result)
cv2.imwrite("output.jpg", result)  # ← 儲存輸出結果
print("已儲存結果為 output.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()