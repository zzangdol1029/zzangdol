import cv2 as cv
import numpy as np
import time


# 직접 작성한 RGB -> Grayscale 함수 (버전 1)
def my_cvtGray1(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r, c] = 0.114 * bgr_img[r, c, 0] + 0.587 * bgr_img[r, c, 1] + 0.299 * bgr_img[r, c, 2]
    return np.uint8(g)


# 직접 작성한 RGB -> Grayscale 함수 (벡터화 활용, 버전 2)
def my_cvtGray2(bgr_img):
    g = 0.114 * bgr_img[:, :, 0] + 0.587 * bgr_img[:, :, 1] + 0.299 * bgr_img[:, :, 2]
    return np.uint8(g)


# 테스트용 이미지 로드
img = cv.imread('../program4/girl_laughing.jpg')
if img is None:
    raise ValueError("이미지 파일을 찾을 수 없습니다. 'girl_laughing.png' 경로를 확인하세요.")

# 실행 시간 측정: my_cvtGray1
start = time.time()
my_cvtGray1(img)
print('My time1:', time.time() - start)

# 실행 시간 측정: my_cvtGray2
start = time.time()
my_cvtGray2(img)
print('My time2:', time.time() - start)

# 실행 시간 측정: OpenCV 내장 함수
start = time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('OpenCV time:', time.time() - start)
