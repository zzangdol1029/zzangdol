import cv2 as cv
import numpy as np

# 영상 읽기
img = cv.imread('../program4/soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 에지 검출
canny = cv.Canny(gray, 100, 200)

# 윤곽선 검출
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# 길이가 100 이상인 윤곽선만 저장
lcontour = []
for i in range(len(contour)):
    if contour[i].shape[0] > 100:
        lcontour.append(contour[i])

# 윤곽선 그리기 (녹색, 두께 3)
cv.drawContours(img, lcontour, -1, (0, 255, 0), 3)

# 결과 출력
cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)

cv.waitKey(0)
cv.destroyAllWindows()
