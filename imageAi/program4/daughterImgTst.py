import cv2
import os

# 이미지 파일 경로 (실제 이미지 경로로 수정 필요)
image_path = '../program4/daughter.png'  # 실제 PNG 파일 경로를 지정
output_path = 'sharpened_image.jpg'

# 경로 확인
if not os.path.exists(image_path):
    print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
    exit(1)

# 이미지 불러오기
image = cv2.imread(image_path)

if image is None:
    print(f"오류: 이미지를 불러올 수 없습니다: {image_path}")
    exit(1)

# 1. 가우시안 블러링 적용
gaussian = cv2.GaussianBlur(image, (5, 5), 5.0)

# 2. 가중치를 강화한 Unsharp Masking
sharpened = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)

# 3. 추가 샤프닝 필터 (커널 사용)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened_final = cv2.filter2D(sharpened, -1, kernel)

# 결과 저장
cv2.imwrite(output_path, sharpened_final)
print(f"이미지가 매우 선명하게 처리되어 '{output_path}'로 저장되었습니다.")
