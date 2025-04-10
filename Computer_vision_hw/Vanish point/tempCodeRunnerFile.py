        max_width = 1280
        max_height = 720
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_size)