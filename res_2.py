# 处理视频，动态追踪
# 进一步调整 形态学操作的参数
import cv2
import numpy as np

# 准备工作
p1_l = (350, 405)  # 上方血管 左侧极值点,用来确定具体血管轮廓
p2_l = (612, 739)  # 下方血管 左侧极值点,用来确定具体血管轮廓

# 使用cv2.VideoCapture读取视频
cap = cv2.VideoCapture("video/test1.mp4")

while cap.isOpened():
    # 读取视频中的图片
    ret, img = cap.read()
    # cv2.imshow("img",img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)

    # 高斯模糊+二值化
    blurred = cv2.GaussianBlur(gray, (61, 61), 0)  # 61
    (_, thresh1) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)  # 90
    (_, thresh2) = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)  # 45

    imgs_1 = np.hstack((blurred, thresh1, thresh2))
    # cv2.imshow("imgs_61;90,45",imgs_1)

    # 形态学处理，ELLIPSE核+Open + CLOSE操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 25
    open1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 15
    open2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 25
    close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernel2)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 15
    close2 = cv2.morphologyEx(open2, cv2.MORPH_CLOSE, kernel3)

    imgs_mor1 = np.hstack((open1, close1))
    imgs_mor2 = np.hstack((open2, close2))
    imgs_mor = np.vstack((imgs_mor1, imgs_mor2))
    # cv2.imshow("morphology",imgs_mor)

    # 膨胀
    dilate1 = cv2.dilate(close1, None, iterations=4)#4
    dilate2 = cv2.dilate(close2, None, iterations=4)#4

    imgs_dia1 = np.hstack((close1, dilate1))
    imgs_dia2 = np.hstack((close2, dilate2))
    imgs_dia = np.vstack((imgs_dia1, imgs_dia2))
    # cv2.imshow("dilate",imgs_dia)

    # 检测轮廓
    # cv2 返回2个值：contours,hierarchy
    # cv3 返回3个值：img，contours，hierarchy
    contours1, hierarchy1 = cv2.findContours(dilate1.copy(),
                                             cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours1))

    contours2, hierarchy2 = cv2.findContours(dilate2.copy(),
                                             cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours2))

    # 通过血管左侧位置，确定上方血管轮廓
    dst1 = []  # 存储每个轮廓左侧极值点距离标准位置p1_l的棋盘距离
    for pentagram in contours1:
        leftmost = tuple(pentagram[:, 0][pentagram[:, :, 0].argmin()])
        d = abs(leftmost[0] - p1_l[0]) + abs(leftmost[1] - p1_l[1])
        dst1.append(d)
    indx1 = dst1.index(min(dst1))

    # 通过血管左侧位置，确定下方血管轮廓
    dst2 = []  # 存储每个轮廓左侧极值点距离标准位置p2_l的棋盘距离
    for pentagram in contours2:
        leftmost = tuple(pentagram[:, 0][pentagram[:, :, 0].argmin()])
        d = abs(leftmost[0] - p2_l[0]) + abs(leftmost[1] - p2_l[1])
        dst2.append(d)
    indx2 = dst2.index(min(dst2))

    # 分别绘制轮廓
    draw_img1 = cv2.drawContours(img.copy(), contours1, indx1, (0, 0, 255), 3)
    draw_img2 = cv2.drawContours(img.copy(), contours2, indx2, (0, 0, 255), 3)

    imgs_draw = np.hstack((draw_img1, draw_img2))
    # cv2.imshow("draw_img", imgs_draw)

    # 绘制最终结果
    res_img = cv2.drawContours(draw_img1.copy(), contours2, indx2, (0, 0, 255), 3)
    cv2.imshow("res_img", res_img)

    # 当视频播放完，或者键盘输入“q”，结束循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()