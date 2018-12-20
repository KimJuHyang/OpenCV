#-*-coding:utf-8-*-

# KIMJUHYANG
# 객체 모양 판별
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label


imgs = cv2.imread("shape1.bmp")

resultIMG = imgs.copy()

#배경과 전체적인 노이즈 흐릿하게하여 배경글자 & 불필요한 선들을 뭉갠다. 도형의 형체만 남긴다.
resultIMG = cv2.medianBlur(resultIMG,3)
resultIMG = cv2.GaussianBlur(resultIMG,(5,5),0)
cv2.imshow('blur', resultIMG)

#그레이 스케일 변환
gray = cv2.cvtColor(resultIMG,cv2.COLOR_BGR2GRAY)
#이진화
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


#모폴로지 연산하여 인접한 픽셀값 조정
kernel = np.ones((2,2),np.uint8)
close = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)
opening = cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel,iterations=1)

#외곽 찾기
edge = cv2.Canny(opening,100,200)
cv2.imshow('edge', edge)
edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#찾은 외곽 그리기
for i in range (0,len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])

num = 0
ractangle = 0
circle = 0
triangle = 0
etc = 0


#contours 배열을 돌며 외곽과 중심점을 체크하며, 카운트를 올린다. (객체구하기)

for i in range (0,len(contours)):

    cnt = contours[i]
    mmt = cv2.moments(cnt)

    #공간값이 없는경우 멈춘다. contours
    if mmt['m00'] == 0:
        continue

    #무게중심점 구하기
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

    #무게중심점 찍어주기
    cv2.circle(imgs, (cx,cy), 5, (0,255,0), -1)

    #contour 근사화 하기. 곱하는 숫자가 커질수록 근사화 정도 커짐.
    epsilon = 0.044 * cv2.arcLength(cnt, True)

    #꼭짓점의 좌표를 approx에 저장하기.
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(imgs, [approx], 0, (0,0,255),3)

    

#오브젝트를 파악하면 총 개수 증가시킴
all = circle+triangle+ractangle+etc
print("** 오브젝트의 총 개수 : %d 개 **"%all)
print("** 원의 총 개수 : %d 개 **"%circle)
print("** 삼각형의 총 개수 : %d 개 **"%triangle)
print("** 사각형의 총 개수 : %d 개 **"%ractangle)
print("** 기타 오브젝트의 총 개수 : %d 개 **"%etc)



cv2.imshow('result', imgs)
cv2.waitKey(0)
