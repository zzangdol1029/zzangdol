import tensorflow as tf
# 하이퍼파라미터란? 모델 학습 전에 사용자가 직접 설정하는 값으로, 모델의 성능과 학습 방식에 큰 영향을 미침
# 이 파일에서 사용되는 주요 하이퍼파라미터:
# - EPOCHS: 추가 학습 반복 횟수
# - BATCH_SIZE: 추가 학습 배치 크기
# - OPTIMIZER: 최적화 함수
# - LOSS: 손실 함수
# - 데이터 증강 방식: 이미지 변형 종류

# -------------------- 하이퍼파라미터 설정 --------------------
EPOCHS = 7  # 추가 학습 반복 횟수 (더 낮춰서 과적합을 더욱 방지)
BATCH_SIZE = 3  # 추가 학습 배치 크기 (더 낮춰서 업데이트 빈도 증가)
from keras.optimizers import RMSprop
OPTIMIZER = RMSprop()  # 최적화 함수 (RMSprop은 학습률을 자동 조정해주는 최적화 알고리즘)
LOSS = 'sparse_categorical_crossentropy'  # 손실 함수 (다중 클래스 분류에서 정수형 라벨에 적합)

import numpy as np
import cv2
from keras.models import load_model
import os

def preprocess_image_for_training(image_path):
    """
    predict_model.py의 preprocess_image 함수를 활용하여 학습용 전처리
    Args:
        image_path (str): 이미지 파일 경로
    Returns:
        np.ndarray: (28, 28, 1) 형태의 전처리된 이미지
    """
    from predict_model import preprocess_image
    img = preprocess_image(image_path)
    return img[0]  # (1, 28, 28, 1) → (28, 28, 1)

def augment_image(img):
    """
    입력 이미지를 다양한 방식(증강)으로 변형하여 리스트로 반환
    - 좌우 반전, 상하 이동, 다양한 각도 회전, 확대/축소, 밝기 변화, 강한 노이즈 등
    - 데이터 증강(Augmentation)은 소량의 데이터를 여러 개로 늘려 overfitting을 방지하고 일반화 성능을 높임
    Args:
        img (np.ndarray): (28, 28, 1) 형태의 이미지
    Returns:
        List[np.ndarray]: 변형된 이미지들의 리스트
    """
    augmented = [img]
    img2d = img.squeeze()  # (28, 28)
    # 1. 좌우 반전
    augmented.append(cv2.flip(img2d, 1)[..., np.newaxis])
    # 2. 상하 이동 (위/아래로 2픽셀씩 이동)
    for dy in [-2, 2]:
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        shifted = cv2.warpAffine(img2d, M, (28, 28), borderValue=0)
        augmented.append(shifted[..., np.newaxis])
    # 3. 다양한 각도 회전 (-30, -15, 15, 30도)
    for angle in [-30, -15, 15, 30]:
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(img2d, M, (28, 28), borderValue=0)
        augmented.append(rotated[..., np.newaxis])
    # 4. 확대/축소 (0.9배, 1.1배)
    for scale in [0.9, 1.1]:
        zoomed = cv2.resize(img2d, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        zoomed = cv2.resize(zoomed, (28, 28), interpolation=cv2.INTER_LINEAR)
        augmented.append(zoomed[..., np.newaxis])
    # 5. 밝기 변화 (0.7배, 1.3배)
    for alpha in [0.7, 1.3]:
        bright = np.clip(img2d * alpha, 0, 1)
        augmented.append(bright[..., np.newaxis])
    # 6. 더 강한 랜덤 노이즈 (표준편차 0.1, 0.2)
    for std in [0.1, 0.2]:
        noise = np.random.normal(0, std, img2d.shape)
        noisy = np.clip(img2d + noise, 0, 1)
        augmented.append(noisy[..., np.newaxis])
    return augmented

if __name__ == "__main__":
    # 학번 및 모델 경로 설정
    student_id = "2025254011"
    model_path = os.path.join(os.path.dirname(__file__), f"{student_id}.h5")
    # 기존 학습된 모델 불러오기
    model = load_model(model_path)
    # 옵티마이저, 손실함수 등 하이퍼파라미터를 새로 설정하여 compile (추가 학습 오류 방지)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    # 추가 학습할 이미지와 정답 라벨 목록
    # (파일명, 정답 숫자) 형태로 입력
    custom_data = [
        ("2.png", 2),  # 2.png 이미지는 숫자 2로 라벨링
        ("6.png", 6),  # 6.png 이미지는 숫자 6으로 라벨링
        ("9.png", 9)   # 9.png 이미지는 숫자 9로 라벨링
    ]

    x_custom = []  # 전처리된 이미지 데이터 리스트
    y_custom = []  # 정답 라벨 리스트
    for fname, label in custom_data:
        image_path = os.path.join(os.path.dirname(__file__), fname)
        img = preprocess_image_for_training(image_path)
        # 데이터 증강(Augmentation) 적용: 각 이미지를 여러 방식으로 변형하여 학습 데이터 다양성 증가
        augmented_imgs = augment_image(img)
        for aug_img in augmented_imgs:
            x_custom.append(aug_img)
            y_custom.append(label)

    x_custom = np.array(x_custom)  # (N, 28, 28, 1) 형태로 변환 (N: 증강된 전체 이미지 개수)
    y_custom = np.array(y_custom)  # (N,) 형태로 변환

    # 기존 모델에 사용자 손글씨 데이터로 추가 학습
    # epochs, batch_size 등 하이퍼파라미터는 상단에서 설정
    model.fit(x_custom, y_custom, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 학습된 모델을 새로운 파일(2025254011_ft.h5)로 저장
    ft_model_path = os.path.join(os.path.dirname(__file__), f"{student_id}_ft.h5")
    model.save(ft_model_path)
    print(f"모델이 사용자 손글씨(2, 6, 9)로 추가 학습되어 {student_id}_ft.h5로 저장되었습니다.") 