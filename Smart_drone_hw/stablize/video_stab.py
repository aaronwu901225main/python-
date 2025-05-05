import cv2
import numpy as np
import os

def stabilize_video(input_path, output_path='output_stabilized.mp4', use_sift=False, crop_output=True):
    cap = cv2.VideoCapture(input_path)
    ret, ref_frame = cap.read()
    if not ret:
        print("無法讀取影片！")
        return

    h, w = ref_frame.shape[:2]
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # 建立特徵提取器
    if use_sift:
        feature = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        feature = cv2.ORB_create(1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ref_kp, ref_des = feature.detectAndCompute(ref_gray, None)

    # 輸出設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = feature.detectAndCompute(frame_gray, None)

        if des is None or len(kp) < 10:
            print(f"[Frame {frame_id}] 特徵點不足，略過對齊")
            aligned = frame
        else:
            matches = matcher.match(ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) >= 10:
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                aligned = cv2.warpPerspective(frame, H, (w, h))
            else:
                print(f"[Frame {frame_id}] 配對不足，略過對齊")
                aligned = frame

        # 裁切中央區域避免黑邊
        if crop_output:
            margin = 30  # 可調整裁切範圍
            aligned = aligned[margin:h - margin, margin:w - margin]
            aligned = cv2.resize(aligned, (w, h))

        out.write(aligned)

        # Optional 顯示
        # cv2.imshow("Stabilized", aligned)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("影片穩定處理完成！輸出檔案：", output_path)

# -------- 用法 ----------
if __name__ == "__main__":
    input_video = "DJI_0025_W.MP4"  # 換成你要的影片路徑
    stabilize_video(input_video, use_sift=False, crop_output=True)
