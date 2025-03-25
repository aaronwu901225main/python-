import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "image.jpg"  # 替換為你的圖像路徑
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"無法讀取圖片：{image_path}")

output_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1.reshape(4)
    x3, y3, x4, y4 = line2.reshape(4)

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None
    else:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return int(x), int(y)

if lines is not None:
    intersections = []
    valid_lines = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None and 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]:
                intersections.append(pt)
                valid_lines.append(lines[i])
                valid_lines.append(lines[j])

    if intersections:
        intersections_np = np.array(intersections)
        vanish_point = np.mean(intersections_np, axis=0).astype(int)

        distances = [np.linalg.norm(np.mean(line.reshape(2, 2), axis=0) - vanish_point) for line in valid_lines]
        closest_indices = np.argsort(distances)[:2]
        chosen_lines = [valid_lines[idx] for idx in closest_indices]

        for line in chosen_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(output_image, (x1, y1), tuple(vanish_point), (255, 0, 0), 2)
            cv2.line(output_image, (x2, y2), tuple(vanish_point), (255, 0, 0), 2)

        cv2.circle(output_image, tuple(vanish_point), 10, (0, 0, 255), -1)

        # 顯示圖像
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(output_rgb)
        plt.title("Detected Lines and Vanishing Point")
        plt.axis("off")
        plt.show()
    else:
        print("⚠️ 無法找到任何交點，請確認圖片中有明顯的透視線條。")
else:
    print("⚠️ 沒有偵測到任何直線，請確認圖片品質或調整參數。")
