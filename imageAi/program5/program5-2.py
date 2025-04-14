import cv2 as cv

# 이미지 읽기
img = cv.imread('mot_color70.jpg')  # 파일명에 맞게 경로 수정 필요
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 흑백 변환

# SIFT 객체 생성
sift = cv.SIFT_create()

# 키포인트와 디스크립터 검출
kp, des = sift.detectAndCompute(gray, None)

# 키포인트를 이미지 위에 그림
gray = cv.drawKeypoints(
    gray, kp, None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 결과 출력
cv.imshow('sift', gray)
cv.waitKey(0)
cv.destroyAllWindows()
