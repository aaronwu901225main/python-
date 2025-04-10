import cv2
import numpy as np
import random
import math
import os

def weighted_median(data, weights):
    sorted_pairs = sorted(zip(data, weights), key=lambda x: x[0])
    total_weight = sum(weights)
    cumulative_weight = 0
    for value, weight in sorted_pairs:
        cumulative_weight += weight
        if cumulative_weight >= total_weight / 2:
            return value
    return sorted_pairs[-1][0]

def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    # 點(x0,y0) 到線段(x1,y1)-(x2,y2)的距離
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    return numerator / denominator if denominator != 0 else float('inf')

def ransac_intersection(lines, threshold=5.0, iterations=200):
    best_inliers = []
    best_point = None

    for _ in range(iterations):
        l1, l2 = random.sample(lines, 2)
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-6:
            continue

        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

        inliers = []
        for lx1, ly1, lx2, ly2 in lines:
            dist = point_to_line_distance(px, py, lx1, ly1, lx2, ly2)
            if dist < threshold:
                inliers.append((px, py))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_point = (px, py)

    return best_point

def find_vanishing_point(image_path, min_line_length=100, angle_threshold=np.pi/6):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 無法讀取圖片")
        return

    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min_line_length, maxLineGap=30)

    if lines is None:
        print("❌ 未檢測到線段")
        return

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue

        theta = math.atan2(dy, dx)
        filtered_lines.append((x1, y1, x2, y2, theta))

    # 過濾角度類似的線條（可選）
    angle_groups = []
    angle_eps = angle_threshold
    for line in filtered_lines:
        grouped = False
        for group in angle_groups:
            if abs(line[4] - group[0][4]) < angle_eps:
                group.append(line)
                grouped = True
                break
        if not grouped:
            angle_groups.append([line])

    # 從每組挑一條線出來做 RANSAC
    representative_lines = [group[0][:4] for group in angle_groups if len(group) > 0]

    if len(representative_lines) < 2:
        print("❌ 有效線段不足以計算交點")
        return

    vp = ransac_intersection(representative_lines)

    if vp is None:
        print("❌ 無法估計消失點")
        return

    x_vp, y_vp = vp

    # 計算每個交點的距離作為權重（距中心越近權重越高）
    intersections = []
    for i in range(len(representative_lines)):
        for j in range(i + 1, len(representative_lines)):
            x1, y1, x2, y2 = representative_lines[i]
            x3, y3, x4, y4 = representative_lines[j]

            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if abs(denom) < 1e-6:
                continue

            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            intersections.append((px, py))

    if intersections:
        xs = [p[0] for p in intersections]
        ys = [p[1] for p in intersections]
        weights = [1] * len(intersections)  # 每個交點權重都一樣

        x_final = int(weighted_median(xs, weights))
        y_final = int(weighted_median(ys, weights))

        # 畫出代表線段
        for line in representative_lines:
            x1, y1, x2, y2 = line
            dist1 = np.hypot(x1 - x_final, y1 - y_final)
            dist2 = np.hypot(x2 - x_final, y2 - y_final)

            if dist1 < 100 or dist2 < 100:  # 100 是你可以調整的距離閾值
                # 取距離較遠的端點畫到消失點
                if dist1 > dist2:
                    cv2.line(img, (x1, y1), (x_final, y_final), (0, 255, 0), 5)
                else:
                    cv2.line(img, (x2, y2), (x_final, y_final), (0, 255, 0), 5)
            else:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # 畫消失點
        cv2.circle(img, (x_final, y_final), 30, (0, 0, 255), -1)
        print(f"✅ 消失點座標: ({x_final}, {y_final})")


        # 顯示與儲存結果
        # 等比例縮放顯示圖片，避免變形
        max_width = 1280
        max_height = 720
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_size)

        try:
            cv2.imshow("Vanishing Point", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("📷 無法顯示圖像（非GUI環境）")

        output_path = os.path.join(os.path.dirname(image_path), 'result.jpg')
        cv2.imwrite(output_path, img)
        print(f"📁 圖像已儲存為: {output_path}")
    else:
        print("❌ 未找到交點")

if __name__ == "__main__":
    image_path = 'Computer_vision_hw/Vanish point/image.jpg'
    find_vanishing_point(image_path, min_line_length=200, angle_threshold=np.pi/20)
