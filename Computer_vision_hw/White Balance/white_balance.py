import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體（Windows）
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤

def white_balance_1(img, percent=0.01):
    '''
    White Patch 白點假設法
    基本假設：畫面上最亮的一小部分像素是白色，用這些像素做通道增益校正。

    :param img: cv2.imread 讀取的 BGR 圖片
    :param percent: 預設取最亮的前 0.01% 作為白點（可自行調整）
    :return: 白平衡後的圖
    '''
    img = img.copy()
    b, g, r = cv2.split(img)
    m, n = b.shape

    # 把 B, G, R 通道展平成一維
    b_flat = b.flatten()
    g_flat = g.flatten()
    r_flat = r.flatten()

    # 取得每個通道的最大亮度 (取前percent%的最大值)
    num_pixels = m * n
    num_highlight = max(int(num_pixels * percent / 100), 1)

    # 對每個通道排序取最大亮度
    b_max = np.mean(np.sort(b_flat)[-num_highlight:])
    g_max = np.mean(np.sort(g_flat)[-num_highlight:])
    r_max = np.mean(np.sort(r_flat)[-num_highlight:])

    # 計算增益
    gain_b = 255.0 / b_max
    gain_g = 255.0 / g_max
    gain_r = 255.0 / r_max

    # 增益修正
    b = np.clip(b * gain_b, 0, 255).astype(np.uint8)
    g = np.clip(g * gain_g, 0, 255).astype(np.uint8)
    r = np.clip(r * gain_r, 0, 255).astype(np.uint8)

    balance_img = cv2.merge([b, g, r])
    return balance_img
 
def white_balance_2(img_input):
    '''
    完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r
 
    return img
 
def white_balance_3(img,gain=1.0):
    '''
    灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)*gain
    Ga = (G * Kg)*gain
    Ra = (R * Kr)*gain

    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]
 
    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img
 
 
def white_balance_4(img):
    '''
    基于图像分析的偏色检测及颜色校正方法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
 
    def detection(img):
        '''计算偏色值'''
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b, M_a, M_b = 0, 0, 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
        D = np.sqrt((np.square(d_a) + np.square(d_b)))
 
        for i in range(m):
            for j in range(n):
                M_a = np.abs(a[i][j] - d_a - 128) + M_a
                M_b = np.abs(b[i][j] - d_b - 128) + M_b
 
        M_a, M_b = M_a / (m * n), M_b / (m * n)
        M = np.sqrt((np.square(M_a) + np.square(M_b)))
        k = D / M
        print('偏色值:%f' % k)
        return
 
    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    # detection(img)
 
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(b[0][0]), int(g[0][0])
    for i in range(m):
        for j in range(n):
            I_r_2[i][j] = int(r[i][j] ** 2)
            I_b_2[i][j] = int(b[i][j] ** 2)
            sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
            sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
            sum_I_g = g[i][j] + sum_I_g
            sum_I_r = r[i][j] + sum_I_r
            sum_I_b = b[i][j] + sum_I_b
            if max_I_r < r[i][j]:
                max_I_r = r[i][j]
            if max_I_r_2 < I_r_2[i][j]:
                max_I_r_2 = I_r_2[i][j]
            if max_I_g < g[i][j]:
                max_I_g = g[i][j]
            if max_I_b_2 < I_b_2[i][j]:
                max_I_b_2 = I_b_2[i][j]
            if max_I_b < b[i][j]:
                max_I_b = b[i][j]
 
    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    # print(u_b, v_b, u_r, v_r)
    b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
            g0[i][j] = g[i][j]
            # r0[i][j] = r[i][j]
            r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
            if r_point>255:
                r0[i][j] = 255
            else:
                if r_point<0:
                    r0[i][j] = 0
                else:
                    r0[i][j] = r_point
            if b_point>255:
                b0[i][j] = 255
            else:
                if b_point<0:
                    b0[i][j] = 0
                else:
                    b0[i][j] = b_point
    return cv2.merge([b0, g0, r0])
 
def white_balance_5(img):
    '''
    动态阈值算法
    算法分为两个步骤：白点检测和白点调整。
    只是白点检测不是与完美反射算法相同的认为最亮的点为白点，而是通过另外的规则确定
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
 
    b, g, r = cv2.split(img)
    """
    YUV空间
    """
    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]
 
    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)
 
    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # 如果该值过大过小，色温向两极端发展
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0
 
            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.1 * ysum:    # 取前10%的亮点为计算值，如果该值过大易过曝光，该值过小调整幅度小
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1
 
    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb
 
    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            if b_point>255:
                b[i][j] = 255
            else:
                if b_point<0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point>255:
                g[i][j] = 255
            else:
                if g_point<0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point>255:
                r[i][j] = 255
            else:
                if r_point<0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point
 
    return cv2.merge([b, g, r])
 
'''
img : 原图
img1：均值白平衡法
img2: 完美反射
img3: 灰度世界假设
img4: 基于图像分析的偏色检测及颜色校正方法
img5: 动态阈值算法
'''
def process_white_balance(image_path, save_results=True, show_results=True, output_dir="white_balance_results"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片，請檢查路徑")
        return

    img1 = white_balance_1(img)
    img2 = white_balance_2(img)
    img3 = white_balance_3(img, gain=1.0) # 調整增益值可改變白平衡效果
    img4 = white_balance_4(img)
    img5 = white_balance_5(img)

    result_imgs = [img, img1, img2, img3, img4, img5]
    result_titles = [
        "原圖(Original)",
        "方法1:White Patch Retinex",
        "方法2:完美反射法",
        "方法3:灰度世界假設",
        "方法4:偏色分析校正",
        "方法5:動態閾值調整法"
    ]
    save_filenames = [
    "original",
    "wb1_White_Patch_Retinex",
    "wb2_perfect_reflect",
    "wb3_gray_world",
    "wb4_color_bias_correction",
    "wb5_dynamic_threshold"
    ]
    # 擷取/後的圖片名並去除.jpg
    for i in image_path:
        if i == '/':
            image_name = image_path.split('/')[-1]
            image_name = image_name.split('.')[0]
            break
        else:
            image_name = image_path.split('\\')[-1]
            image_name = image_name.split('.')[0]

    # 儲存圖片
    if save_results:
        for i, im in enumerate(result_imgs):
            cv2.imwrite(os.path.join(output_dir, f"{save_filenames[i]}_({image_name}).jpg"), im)

    # 顯示圖片
    if show_results:
        plt.figure(figsize=(12, 8))
        for i, im in enumerate(result_imgs):
            rgb_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 3, i + 1)
            plt.imshow(rgb_img)
            plt.title(result_titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# === 主程式入口 ===
if __name__ == "__main__":
    process_white_balance(
        image_path=r'./Computer_vision_hw/White Balance/white-balance-auto-sample-image_1465-1.jpg',
        save_results=True,
        show_results=True,
        output_dir=r"./Computer_vision_hw/White Balance/white_balance_results"
    )
