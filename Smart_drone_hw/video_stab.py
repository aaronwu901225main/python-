import cv2
import numpy as np
import math

# 穩定半徑，越大代表平滑程度越高，但對移動的反應變慢
SMOOTHING_RADIUS = 1000
# 邊界裁剪，以避免黑邊明顯
HORIZONTAL_BORDER_CROP = 20

# 定義平移與旋轉變換的參數
class TransformParam:
    def __init__(self, dx=0, dy=0, da=0):
        self.dx = dx
        self.dy = dy
        self.da = da  # 旋轉角度

# 定義軌跡（累積的位移和角度）
class Trajectory:
    def __init__(self, x=0, y=0, a=0):
        self.x = x
        self.y = y
        self.a = a

# 主程序開始
def video_stabilization(video_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "無法開啟影片"

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 讀取第一幀並轉灰階
    ret, prev = cap.read()
    if not ret:
        print("無法讀取影片第一幀")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # 初始化輸出影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('stabilized_output.mp4', fourcc, fps, (prev.shape[1], prev.shape[0]))

    prev_to_cur_transform = []  # 所有幀的變換參數
    last_T = np.eye(2, 3, dtype=np.float64)

    for k in range(1, n_frames):
        success, cur = cap.read()
        if not success:
            break

        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # 使用光流追蹤特徵點
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None)

        # 過濾掉失敗的匹配
        valid_prev = prev_pts[status.flatten() == 1]
        valid_cur = cur_pts[status.flatten() == 1]

        # 使用剛性轉換估計（不包含縮放與剪切）
        T = cv2.estimateAffinePartial2D(valid_prev, valid_cur)[0]

        if T is None:
            T = last_T.copy()
        last_T = T.copy()

        dx = T[0, 2]
        dy = T[1, 2]
        da = math.atan2(T[1, 0], T[0, 0])

        prev_to_cur_transform.append(TransformParam(dx, dy, da))

        prev_gray = cur_gray

        print(f"處理第 {k}/{n_frames} 幀，有效特徵點: {len(valid_prev)}")

    # 累積所有變換得到原始軌跡
    trajectory = []
    x = y = a = 0
    for p in prev_to_cur_transform:
        x += p.dx
        y += p.dy
        a += p.da
        trajectory.append(Trajectory(x, y, a))

    # 對軌跡進行平滑處理
    smoothed_trajectory = []
    for i in range(len(trajectory)):
        sum_x = sum_y = sum_a = count = 0
        for j in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS + 1):
            idx = i + j
            if 0 <= idx < len(trajectory):
                sum_x += trajectory[idx].x
                sum_y += trajectory[idx].y
                sum_a += trajectory[idx].a
                count += 1
        smoothed_trajectory.append(Trajectory(sum_x / count, sum_y / count, sum_a / count))

    # 計算新轉換，使結果符合平滑後的軌跡
    new_prev_to_cur_transform = []
    x = y = a = 0
    for i in range(len(prev_to_cur_transform)):
        x += prev_to_cur_transform[i].dx
        y += prev_to_cur_transform[i].dy
        a += prev_to_cur_transform[i].da

        diff_x = smoothed_trajectory[i].x - x
        diff_y = smoothed_trajectory[i].y - y
        diff_a = smoothed_trajectory[i].a - a

        dx = prev_to_cur_transform[i].dx + diff_x
        dy = prev_to_cur_transform[i].dy + diff_y
        da = prev_to_cur_transform[i].da + diff_a

        new_prev_to_cur_transform.append(TransformParam(dx, dy, da))

    # 套用新的轉換到影片中
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vert_border = HORIZONTAL_BORDER_CROP * prev.shape[0] // prev.shape[1]

    for k in range(len(new_prev_to_cur_transform)):
        success, frame = cap.read()
        if not success:
            break

        dx = new_prev_to_cur_transform[k].dx
        dy = new_prev_to_cur_transform[k].dy
        da = new_prev_to_cur_transform[k].da

        T = np.array([
            [math.cos(da), -math.sin(da), dx],
            [math.sin(da),  math.cos(da), dy]
        ])

        frame_stabilized = cv2.warpAffine(frame, T, (frame.shape[1], frame.shape[0]))

        # 裁剪邊界，避免黑邊
        frame_stabilized = frame_stabilized[vert_border:-vert_border,
                                            HORIZONTAL_BORDER_CROP:-HORIZONTAL_BORDER_CROP]
        frame_stabilized = cv2.resize(frame_stabilized, (frame.shape[1], frame.shape[0]))
        out.write(frame_stabilized)
        
        # 顯示前後畫面比較
        canvas = np.zeros((frame.shape[0], frame.shape[1]*2 + 10, 3), dtype=np.uint8)
        canvas[:, :frame.shape[1]] = frame
        canvas[:, frame.shape[1]+10:] = frame_stabilized

        if canvas.shape[1] > 1920:
            canvas = cv2.resize(canvas, (canvas.shape[1]//2, canvas.shape[0]//2))

        cv2.imshow("Before and After", canvas)
        if cv2.waitKey(20) == 27:  # 按 ESC 退出
            break

    cap.release()
    out.release()
    print("影片穩定化完成，輸出檔案: stabilized_output.avi")
    cv2.destroyAllWindows()

# 執行主程序
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python video_stab.py [影片路徑]")
    else:
        video_stabilization(sys.argv[1])
