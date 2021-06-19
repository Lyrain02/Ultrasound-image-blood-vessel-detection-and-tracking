# 对视频中截取的图片进行检测
# 利用高斯模糊+二值化+形态学处理+轮廓检测，确定血管位置
# res_1只能跑通2.png
import cv2
import numpy as np

# 读取图像+灰度化
img_path = r'images/2.png'
img = cv2.imread(img_path) #cv2读进来的图片是BGR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("img",img)
# cv2.imshow("gray",gray)

# 高斯模糊+二值化
blurred = cv2.GaussianBlur(gray, (61, 61),0)
(_, thresh1) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV) # 上方血管，90
(_, thresh2) = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV) # 下方血管，45

# imgs_1 = np.hstack((blurred,thresh1,thresh2))
# cv2.imshow("imgs_61;90,45",imgs_1)


# 形态学处理，ELLIPSE核+Open + CLOSE操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))#30
open1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
open2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (42, 42)) #42
close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernel2)
close2 = cv2.morphologyEx(open2, cv2.MORPH_CLOSE, kernel2)

imgs_mor1 = np.hstack((open1,close1))
imgs_mor2 = np.hstack((open2,close2))
imgs_mor = np.vstack((imgs_mor1,imgs_mor2))
# cv2.imshow("morphology",imgs_mor)
# cv2.imshow("morphology_up",imgs_mor1)
# cv2.imshow("morphology_down",imgs_mor2)

# 膨胀
dilate1 = cv2.dilate(close1, None, iterations=4)
dilate2 = cv2.dilate(close2, None, iterations=4)

imgs_dia1 = np.hstack((close1,dilate1))
imgs_dia2 = np.hstack((close2,dilate2))
imgs_dia = np.vstack((imgs_dia1,imgs_dia2))
# cv2.imshow("dilate",imgs_dia)
# cv2.imshow("dilate_up",imgs_dia1)
# cv2.imshow("dilate_down",imgs_dia2)

# 检测轮廓
# cv2 返回2个值：contours,hierarchy
# cv3 返回3个值：img，contours，hierarchy
contours1, hierarchy1 = cv2.findContours(dilate1.copy(),
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours1))

contours2, hierarchy2 = cv2.findContours(dilate2.copy(),
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours2))

# 分别绘制轮廓
draw_img1 = cv2.drawContours(img.copy(), contours1, 0, (0, 0, 255), 3)
draw_img2 = cv2.drawContours(img.copy(), contours2, 0, (0, 0, 255), 3)

# imgs_draw = np.hstack((draw_img1,draw_img2))
# cv2.imshow("draw_img", imgs_draw)

# 绘制最终结果
res_img = cv2.drawContours(img.copy(), contours1, 0, (0, 0, 255), 3)
res_img = cv2.drawContours(res_img.copy(), contours2, 0, (0, 0, 255), 3)
cv2.imshow("res_img", res_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("finish")