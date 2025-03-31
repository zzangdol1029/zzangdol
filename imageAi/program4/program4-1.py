import cv2 as cv

# 이미지 읽기
img = cv.imread('../program4/soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel 연산자 적용 (X, Y 방향)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# 절댓값 취한 후 CV_8U로 변환 (0~255 사이 정수)
sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

# 두 방향의 에지 합성 (가중치 평균)
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# 결과 출력
cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey(0)
cv.destroyAllWindows()
