import cv2 as cv
import numpy as np

# 이미지 읽기 및 복사본 생성 (표시용)
img = cv.imread('../program4/soccer.jpg')         # 영상 읽기
img_show = np.copy(img)                           # 붓칠을 디스플레이할 목적의 영상

# GrabCut에서 사용할 마스크 초기화 (전부 배경으로 설정)
mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:, :] = cv.GC_PR_BGD                         # 모든 화소를 '가능성 있는 배경'으로 초기화

# 붓 크기와 색상 설정
BrushSiz = 9                                      # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)         # 파란색(물체)과 빨간색(배경)

# 마우스 이벤트로 붓칠하는 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # 왼쪽 버튼 클릭하면 파란색 (물체로 인식)
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        # 오른쪽 버튼 클릭하면 빨간색 (배경으로 인식)
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        # 왼쪽 버튼 누른 채 이동: 파란색으로 계속 표시
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        # 오른쪽 버튼 누른 채 이동: 빨간색으로 계속 표시
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)

# 윈도우 생성 및 마우스 콜백 함수 연결
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

# 사용자 붓칠 입력 루프 (q 키 누르면 종료)
while True:
    cv.imshow('Painting', img_show)
    if cv.waitKey(1) == ord('q'):
        break

# -------------------------------
# 여기부터 GrabCut 적용하는 코드
# -------------------------------

# GrabCut에서 사용하는 히스토그램 초기화
background = np.zeros((1, 65), np.float64)        # 배경 모델
foreground = np.zeros((1, 65), np.float64)        # 물체 모델

# GrabCut 수행 (마스크 기반 초기화 방식)
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)

# 최종 결과 마스크 생성: 확정 또는 가능성 있는 물체(1), 나머지는 배경(0)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 마스크 적용하여 최종 결과 추출
grab = img * mask2[:, :, np.newaxis]

# 결과 이미지 출력
cv.imshow('Grab cut image', grab)

cv.waitKey()
cv.destroyAllWindows()
