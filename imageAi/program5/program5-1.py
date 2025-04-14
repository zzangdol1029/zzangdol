import cv2 as cv
import numpy as np

# 샘플 이미지 행렬 (8x8 이미지)
img = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
], dtype=np.float32)

# Sobel 필터처럼 방향 필터 정의
ux = np.array([[-1, 0, 1]])
uy = np.array([[-1], [0], [1]])

# 가우시안 커널 생성 (3x3)
k = cv.getGaussianKernel(3, 1)
g = np.outer(k, k.transpose())

# x, y 방향 미분
dy = cv.filter2D(img, cv.CV_32F, uy)
dx = cv.filter2D(img, cv.CV_32F, ux)

# 제곱 및 곱
dyy = dy * dy
dxx = dx * dx
dyx = dy * dx

# 가우시안 가중치 적용
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)

# 해리스 응답 계산
C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) ** 2

# 극점 검출 (비최대 억제 + 임계값)
for j in range(1, C.shape[0] - 1):
    for i in range(1, C.shape[1] - 1):
        if C[j, i] > 0.1 and sum(C[j, i] > C[j-1:j+2, i-1:i+2].flatten()) == 8:
            img[j, i] = 9  # 특징점을 원본 이미지에 9로 표시

# 결과 출력용
np.set_printoptions(precision=2)
print("dy:\n", dy)
print("dx:\n", dx)
print("dyy:\n", dyy)
print("dxx:\n", dxx)
print("dyx:\n", dyx)
print("gdyy:\n", gdyy)
print("gdxx:\n", gdxx)
print("gdyx:\n", gdyx)
print("C (해리스 응답):\n", C)
print("img (결과):\n", img)

# 시각화용 이미지 (16배 확대)
popping = np.zeros((160, 160), np.uint8)
for j in range(160):
    for i in range(160):
        popping[j, i] = np.uint8((C[j // 16, i // 16] + 0.06) * 700)

cv.imshow('Image Display2', popping)
cv.waitKey(0)
cv.destroyAllWindows()
