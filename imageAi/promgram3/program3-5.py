import cv2 as cv
import numpy as np

# 이미지 읽고 크기 조정
img = cv.imread('../program4/soccer.jpg')
img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)


# 감마 함수 정의
def gamma(f, gamma=1.0):
    f1 = f / 255.0
    return np.uint8(255.0 * (f1 ** (1.0 / gamma)))


# 서로 다른 감마 값을 적용한 결과 생성
gc = np.hstack((gamma(img, 0.5), gamma(img, 0.75), gamma(img, 1.0), gamma(img, 2.0), gamma(img, 3.0)))

# 결과 출력
cv.imshow('gamma', gc)
cv.waitKey(0)
cv.destroyAllWindows()
