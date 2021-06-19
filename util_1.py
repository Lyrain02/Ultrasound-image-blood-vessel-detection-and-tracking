# 以 res_1.py + 2.png为基准，确定上、下血管的标准位置
# 为后续使用位置因素精化模型提供参考

import cv2
import numpy as np

img_path = r'images/2.png'
img = cv2.imread(img_path) #cv2读进来的图片是BGR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊+二值化
blurred = cv2.GaussianBlur(gray, (61, 61),0) #61
(_, thresh1) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV) #90
(_, thresh2) = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV) #45

imgs_1 = np.hstack((blurred,thresh1,thresh2))
# cv2.imshow("imgs_61;90,45",imgs_1)


# 形态学处理，ELLIPSE核+Open + CLOSE操作 填充内部细节，此处不能使用Open操作
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

# 膨胀
dilate1 = cv2.dilate(close1, None, iterations=4)
dilate2 = cv2.dilate(close2, None, iterations=4)

imgs_dia1 = np.hstack((close1,dilate1))
imgs_dia2 = np.hstack((close2,dilate2))
imgs_dia = np.vstack((imgs_dia1,imgs_dia2))
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

# 找到上方血管的水平极值点
# (350, 405) (433, 424)
pentagram = contours1[0]
leftmost = tuple(pentagram[:, 0][pentagram[:, :, 0].argmin()])
rightmost = tuple(pentagram[:, 0][pentagram[:, :, 0].argmax()])
print("leftmost",leftmost)
print("rightmost",rightmost)
res1 = cv2.circle(img.copy(), leftmost, 2, (0, 255, 0), 3)
res1 = cv2.circle(res1.copy(), rightmost, 2, (0, 0, 255), 3)
cv2.imshow("res1", res1)

# 找到下方血管的水平极值点
# (612, 739) (664, 746)
pentagram2 = contours2[0]
leftmost2 = tuple(pentagram2[:, 0][pentagram2[:, :, 0].argmin()])
rightmost2 = tuple(pentagram2[:, 0][pentagram2[:, :, 0].argmax()])
print("leftmost2",leftmost2)
print("rightmost2",rightmost2)
res2 = cv2.circle(img.copy(), leftmost2, 2, (0, 255, 0), 3)
res2 = cv2.circle(res2.copy(), rightmost2, 2, (0, 0, 255), 3)
cv2.imshow("res2", res2)


cv2.waitKey(0)
cv2.destroyAllWindows()
print("finish")