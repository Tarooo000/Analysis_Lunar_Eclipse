# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:02:58 2019

@author: 山中　孝太郎
"""

import cv2
import numpy as np

imgname = 'img'
img = cv2.imread(imgname + '.jpg')
B, G, R = cv2.split(img)

#円検出と領域抽出
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 10000, param1=50, param2=20, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

trimb = B[circles[0,0,1] - 1582 : circles[0,0,1] + 1582, circles[0,0,0] - 1582 : circles[0,0,0] + 1582]
trimrgb = img[circles[0,0,1] - 1582 : circles[0,0,1] + 1582, circles[0,0,0] - 1582 : circles[0,0,0] + 1582]

#本影の輪郭の抽出
black = np.zeros((3164, 3164), np.uint8)

sikii = 3
ret, thresh = cv2.threshold(trimb, sikii, 255, cv2.THRESH_BINARY)

#openingとclosingをやってもいいかもしれない
"""
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((4,4),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
"""

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
maxarea = 0
maxindex = 0
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > maxarea:
        maxarea = area
        maxindex = i
    else:
        pass
edge = cv2.drawContours(black, contours, maxindex, (255, 255, 255), 1)
edge = cv2.circle(edge, (1582, 1582), 1540, (0, 0, 0), 30)

#本影の輪郭の座標を抽出
edgepoint = list(np.where(edge == 255))
xn = [x for x in edgepoint[1]]
yn = [y for y in edgepoint[0]]
edgepointlist = np.array([xn, yn])
edgepointlist = edgepointlist.transpose()

#座標をtextで書き出し（必要なし）
"""
file = open(imgname + '.txt', 'w')
for xy in edgepointlist:
    file.write(str(xy[0]) + ',' + str(xy[1]) + '\n')
file.close
"""

#本影の座標を円近似

x = edgepointlist[:, 0]
y = edgepointlist[:, 1]

x2 = x**2
y2 = y**2
x3 = x**3
y3 = y**3
xy = x*y
x2y = x2*y
xy2 = x*y2

xsum = np.sum(x)
ysum = np.sum(y)
x2sum = np.sum(x2)
y2sum = np.sum(y2)
x3sum = np.sum(x3)
y3sum = np.sum(y3)
xysum = np.sum(xy)
x2ysum = np.sum(x2y)
xy2sum = np.sum(xy2)

l = len(x)

a = -x3sum - xy2sum
b = -x2ysum - y3sum
c = -x2sum - y2sum

list1 = np.array([[x2sum, xysum, xsum], [xysum, y2sum, ysum], [xsum, ysum, l]])
list1_inv = np.linalg.inv(list1)
list2 = np.array([[a], [b], [c]])

sol = np.dot(list1_inv, list2)

cx = -sol[0, 0]/2
cy = -sol[1, 0]/2
r = np.sqrt(-sol[2, 0] + cx**2 + cy**2)

#結果の書き出し
result = cv2.drawContours(trimrgb, contours, -1, (255, 0, 255), 2)
result = cv2.circle(result, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 5)

cv2.imwrite('r-' + imgname + '.jpg', trimrgb)
print(r)