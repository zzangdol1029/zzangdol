import skimage
import numpy as np
import cv2 as cv

# 말 이미지 불러오기 (흑백 silhouette 형태)
orig = skimage.data.horse()

# 이진 이미지를 0-255 범위의 반전된 8bit 이미지로 변환
img = 255 - np.uint8(orig) * 255
cv.imshow('Horse', img)

# 외곽 윤곽선 검출 (가장 바깥 윤곽만 추출)
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 컬러 디스플레이용 영상으로 변환
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # 컬러 디스플레이용 영상

# 전체 윤곽선 그리기 (보라색, 두께 2)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow('Horse with contour', img2)

# 가장 큰 외곽선만 사용
contour = contours[0]

# ---------------------------------------------------------
# 윤곽선 기반 특징 계산 및 시각화
# ---------------------------------------------------------

# 모멘트(중심좌표, 면적 등)를 계산
m = cv.moments(contour)

# 면적 계산
area = cv.contourArea(contour)

# 무게중심 좌표 계산 (1차 모멘트 / 0차 모멘트)
cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']

# 윤곽선의 둘레 계산
perimeter = cv.arcLength(contour, True)

# 동근 정도(원형성) 계산: 4π * 면적 / 둘레^2 → 원이면 1에 가까움
roundness = (4.0 * np.pi * area) / (perimeter * perimeter)

# 결과 출력
print('면적=', area, '\n중점=(', cx, ',', cy, ')', '\n둘레=', perimeter, '\n둥근 정도=', roundness)

# ---------------------------------------------------------
# 근사 윤곽선 및 볼록 껍질(convex hull) 시각화
# ---------------------------------------------------------

# 컬러 영상 복사본 생성
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # 컬러 디스플레이용 영상

# 직선 근사화 (정밀도=8픽셀)
contour_approx = cv.approxPolyDP(contour, 8, True)  # 직선 근사
cv.drawContours(img3, [contour_approx], -1, (0, 255, 0), 2)  # 녹색 윤곽

# 볼록 껍질 구하기
hull = cv.convexHull(contour)  # 볼록 껍질

# 모양 맞추기 위해 reshape
hull = hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255), 2)  # 빨간 윤곽

# 최종 시각화
cv.imshow('Horse with line segments and convex hull', img3)

cv.waitKey()
cv.destroyAllWindows()
