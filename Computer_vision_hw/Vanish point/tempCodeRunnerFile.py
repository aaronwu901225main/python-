import cv2
import numpy as np

def find_vanishing_point(image_path, min_line_length=1000, angle_threshold=np.pi/1000):  # 預 增加角度閾值參數
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片")
        return
    
    height, width = img.shape[:2]  # 獲取圖片尺寸
    
    # 轉換成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 邊緣檢測
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 使用 HoughLinesP 檢測線段
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300, minLineLength=min_line_length, maxLineGap=1000)
    
    if lines is None:
        print("未檢測到線段")
        return
    
    # 將線段轉換為極坐標形式並過濾
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < min_line_length:
            continue
            
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        theta = np.arctan2(dy, dx)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        
        # 角度篩選
        if not filtered_lines:
            filtered_lines.append((rho, theta, (x1, y1, x2, y2)))
        else:
            # 檢查新線的角度與現有線的角度差異
            min_angle_diff = min(abs(theta - t) for _, t, _ in filtered_lines)
            if min_angle_diff > angle_threshold:  # 角度差異大於閾值才加入
                filtered_lines.append((rho, theta, (x1, y1, x2, y2)))
    
    if len(filtered_lines) < 2:
        print("有效線段不足以計算交點")
        return
    
    # 計算交點並畫線
    intersections = []
    for i in range(len(filtered_lines)):
        rho1, theta1, (x1, y1, x2, y2) = filtered_lines[i]
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for j in range(i + 1, len(filtered_lines)):
            rho2, theta2, _ = filtered_lines[j]
            a2 = np.cos(theta2)
            b2 = np.sin(theta2)
            
            denom = a1 * b2 - a2 * b1
            if abs(denom) > 1e-10:
                x = (b2 * rho1 - b1 * rho2) / denom
                y = (-a2 * rho1 + a1 * rho2) / denom
                intersections.append([x, y])
    
    if intersections:
        # 計算消失點
        vanishing_point = np.median(intersections, axis=0)
        x_vp, y_vp = int(vanishing_point[0]), int(vanishing_point[1])
        
        # 標記消失點（放大圓點）
        if (0 <= x_vp <= width) and (0 <= y_vp <= height):
            cv2.circle(img, (x_vp, y_vp), 50, (0, 0, 255), -1)  
            print(f"消失點座標: ({x_vp}, {y_vp}) - 在圖片內")
        else:
            x_vp_edge = max(0, min(x_vp, width - 1))
            y_vp_edge = max(0, min(y_vp, height - 1))
            cv2.circle(img, (x_vp_edge, y_vp_edge), 50, (0, 0, 255), -1)  
            cv2.putText(img, "  VP outside", (x_vp_edge + 20, y_vp_edge + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            print(f"消失點座標: ({x_vp}, {y_vp}) - 在圖片外")
        
        # 調整顯示大小
        display_img = cv2.resize(img, (min(1280, width), min(720, height)))
        cv2.imshow('Vanishing Point Detection', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('result.jpg', img)
    else:
        print("未找到消失點")

if __name__ == "__main__":
    # 請將 'image.jpg' 替換成你的圖片路徑
    image_path = r'C:\Users\AaronWu\Documents\GitHub\python-\Computer_vision_hw\Vanish point\image.jpg'
    find_vanishing_point(image_path, min_line_length=2000)  # 可調整角度閾值