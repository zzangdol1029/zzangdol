import numpy as np  # 수치 연산을 위한 라이브러리
from keras.datasets import mnist  # MNIST 데이터셋 로드
from keras.models import Sequential  # 순차적 레이어 모델 생성
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten  # 신경망 레이어
import matplotlib.pyplot as plt  # 그래프 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 깨짐 방지
import tensorflow as tf  # 딥러닝 프레임워크
import os  # 파일 시스템 작업

# GPU 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU 사용 가능")
else:
    print("GPU 사용 불가능")

def create_model():
    """
    CNN 모델을 생성하는 함수
    Returns:
        model: 컴파일된 CNN 모델
    """
    model = Sequential([
        # 첫 번째 컨볼루션 레이어: 32개의 3x3 필터 사용
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # 2x2 크기의 최대 풀링 레이어
        MaxPooling2D((2, 2)),
        # 두 번째 컨볼루션 레이어: 64개의 3x3 필터 사용
        Conv2D(64, (3, 3), activation='relu'),
        # 2x2 크기의 최대 풀링 레이어
        MaxPooling2D((2, 2)),
        # 세 번째 컨볼루션 레이어: 128개의 3x3 필터 사용
        Conv2D(128, (3, 3), activation='relu'),
        # 2차원 특징 맵을 1차원으로 평탄화
        Flatten(),
        # 출력 레이어: 10개의 클래스에 대한 확률 출력
        Dense(10, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(optimizer='rmsprop',  # RMSprop 옵티마이저 사용
                 loss='sparse_categorical_crossentropy',  # 손실 함수
                 metrics=['accuracy'])  # 평가 지표
    return model

def train_and_save_model(student_id):
    """
    모델을 학습하고 저장하는 함수
    Args:
        student_id (str): 학번 (파일명으로 사용)
    """
    # 1. 데이터 로드 및 전처리
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # 이미지 데이터 전처리
    # - 28x28 이미지를 28x28x1 형태로 변환 (채널 추가)
    # - 픽셀값을 0~1 사이로 정규화
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    
    # 2. 모델 생성 및 학습
    model = create_model()
    
    # 모델 학습
    history = model.fit(train_images, train_labels, 
                       epochs=10,  # 전체 데이터셋을 10번 반복 학습
                       batch_size=64,  # 한 번에 64개 샘플 처리
                       validation_data=(test_images, test_labels))  # 검증 데이터
    
    # 3. 모델 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"테스트 정확도: {test_acc:.3f}")
    print(f"테스트 손실: {test_loss:.3f}")
    
    # 4. 모델 저장
    model_filename = os.path.join(os.path.dirname(__file__), f"{student_id}.h5")
    model.save(model_filename)
    print(f"모델이 {model_filename}로 저장되었습니다.")
    
    # 5. 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='학습 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('모델 정확도')
    plt.xlabel('에폭')
    plt.ylabel('정확도')
    plt.legend()
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='학습 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('모델 손실')
    plt.xlabel('에폭')
    plt.ylabel('손실')
    plt.legend()
    
    plt.tight_layout()
    graph_filename = os.path.join(os.path.dirname(__file__), f"{student_id}_training_history.png")
    plt.savefig(graph_filename)
    plt.close()

if __name__ == "__main__":
    # 고정된 학번으로 모델 학습 및 저장
    student_id = "2025254011"
    train_and_save_model(student_id) 