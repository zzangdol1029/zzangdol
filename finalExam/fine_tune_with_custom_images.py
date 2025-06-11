import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import RMSprop
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

if __name__ == "__main__":
    # 학번 및 모델 경로 설정
    student_id = "2025254011"
    model_path = os.path.join(os.path.dirname(__file__), f"{student_id}.h5")
    # 기존 학습된 모델 불러오기
    model = load_model(model_path)
    # 옵티마이저를 새로 설정하여 compile (추가 학습 오류 방지)
    model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
        # 이미지 전처리 (MNIST 스타일)
        x_custom.append(preprocess_image_for_training(image_path))
        y_custom.append(label)

    x_custom = np.array(x_custom)  # (3, 28, 28, 1) 형태로 변환
    y_custom = np.array(y_custom)  # (3,) 형태로 변환

    # 기존 모델에 사용자 손글씨 데이터로 추가 학습
    # epochs=1000, batch_size=100로 소량 데이터에 적합하게 설정
    model.fit(x_custom, y_custom, epochs=100, batch_size=10)

    # 학습된 모델을 새로운 파일(2025254011_ft.h5)로 저장
    ft_model_path = os.path.join(os.path.dirname(__file__), f"{student_id}_ft.h5")
    model.save(ft_model_path)
    print(f"모델이 사용자 손글씨(2, 6, 9)로 추가 학습되어 {student_id}_ft.h5로 저장되었습니다.") 