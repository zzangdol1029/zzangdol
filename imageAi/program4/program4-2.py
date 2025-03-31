import cv2 as cv

# 영상 읽기
img = cv.imread('../program4/soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 케니 에지 검출 (임계값 설정)
canny1 = cv.Canny(gray, 50, 150)   # T_low=50, T_high=150
canny2 = cv.Canny(gray, 100, 200)  # T_low=100, T_high=200

# 결과 출력
cv.imshow('Original', gray)
cv.imshow('Canny1', canny1)
cv.imshow('Canny2', canny2)

cv.waitKey(0)
cv.destroyAllWindows()
