import cv2
import numpy as np
import os

def smooth_trajectory(trajectory, radius=30):
    smoothed = np.copy(trajectory)
    for i in range(3):  # x, y, angle
        smoothed[:, i] = cv2.blur(trajectory[:, i].reshape(-1, 1), (radius, 1)).flatten()
    return smoothed

def fix_border(frame):
    s = frame.shape
    scale = 1.04  # Slight zoom to avoid black borders
    T = cv2.getRotationMatrix2D((s[1]//2, s[0]//2), 0, scale)
    return cv2.warpAffine(frame, T, (s[1], s[0]))

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = []

    for _ in range(n_frames - 1):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        da = 0  # No rotation estimation in this basic method

        transforms.append([dx, dy, da])
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(np.array(trajectory))
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    _, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(fix_border(frame))

    for i in range(len(transforms_smooth)):
        success, frame = cap.read()
        if not success:
            break

        dx, dy, da = transforms_smooth[i]
        m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)
        out.write(frame_stabilized)

    cap.release()
    out.release()
    print(f"Stabilized video saved to: {output_path}")

if __name__ == "__main__":
    input_file = "DJI_0025_W.MP4"
    output_file = "DJI_0025_W_stabilized.mp4"
    if os.path.exists(input_file):
        stabilize_video(input_file, output_file)
    else:
        print(f"Input file '{input_file}' not found.")
