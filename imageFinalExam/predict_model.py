import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import os

def remove_horizontal_lines(img):
    """
    이미지에서 수평선(가로줄)을 제거하는 함수
    Args:
        img (np.ndarray): 이진화된 이미지 (숫자는 흰색, 배경은 검정)
    Returns:
        np.ndarray: 수평선이 제거된 이미지
    """
    # 수평선 검출용 커널 (가로로 긴 형태)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 15, 1))
    # 수평선 검출
    detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # 원본에서 수평선 제거
    img_no_lines = cv2.subtract(img, detected_lines)
    return img_no_lines

def preprocess_image(image_path):
    """
    이미지를 MNIST 스타일에 맞게 전처리하는 함수 (컨트라스트 자동 조정, 적응형 이진화, 노이즈 제거, 수평선 제거, 중앙 정렬, 패딩, 숫자 두껍게 등)
    Args:
        image_path (str): 이미지 파일 경로
    Returns:
        preprocessed_image: 전처리된 이미지
    """
    # 1. 그레이스케일 변환 (컬러 이미지를 흑백으로 변환)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # 2. CLAHE(컨트라스트 자동 조정) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 3. 가우시안 블러로 노이즈 제거
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. 적응형 이진화 + 반전 (숫자가 흰색, 배경이 검정)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)

    # 5. Morphology 연산(열림)으로 작은 노이즈 제거
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. 수평선(가로줄) 제거
    img = remove_horizontal_lines(img)

    # 7. 숫자 영역만 크롭 (여백 제거)
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # 8. 정사각형 패딩 (숫자가 중앙에 오도록)
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # 9. 숫자 두껍게 만들기 (dilate, 커널 크기 증가)
    kernel = np.ones((3,3), np.uint8)
    padded = cv2.dilate(padded, kernel, iterations=1)

    # 10. 28x28로 리사이즈 (MNIST와 동일하게)
    img = cv2.resize(padded, (28, 28))

    # 11. 정규화 (0~1 사이 값)
    img = img.astype('float32') / 255.0

    # 12. 모델 입력 형태로 변환 (배치, 높이, 너비, 채널)
    img = img.reshape(1, 28, 28, 1)

    return img

def predict_digit(model_path, image_path, output_path):
    """
    이미지의 숫자를 예측하고 결과를 파일로 저장하는 함수
    Args:
        model_path (str): 학습된 모델 파일 경로
        image_path (str): 예측할 이미지 파일 경로
        output_path (str): 결과를 저장할 텍스트 파일 경로
    """
    try:
        # 모델 로드
        model = load_model(model_path)
        print(f"모델이 로드되었습니다: {model_path}")
        
        # 이미지 전처리
        preprocessed_image = preprocess_image(image_path)
        
        # 예측
        prediction = model.predict(preprocessed_image)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit]
        
        # 결과를 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"예측된 숫자: {predicted_digit}\n")
            f.write(f"신뢰도: {confidence:.4f}\n")
            f.write("\n각 숫자별 확률:\n")
            for i, prob in enumerate(prediction[0]):
                f.write(f"숫자 {i}: {prob:.4f}\n")
        
        print(f"예측 결과가 저장되었습니다: {output_path}")
        print(f"예측된 숫자: {predicted_digit} (신뢰도: {confidence:.4f})")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def split_and_predict_multi_digits(model_path, image_path, output_dir, student_id):
    """
    한 이미지에 여러 숫자가 있을 때, 각 숫자를 분리하여 예측하고 결과를 저장하는 함수
    Args:
        model_path (str): 학습된 모델 파일 경로
        image_path (str): 예측할 이미지 파일 경로
        output_dir (str): 결과를 저장할 디렉토리 경로
        student_id (str): 학번
    """
    # 1. 이미지 읽기 및 전처리 (흑백 변환, 블러, 이진화)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 숫자 영역 찾기 (contour)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x좌표 기준으로 정렬 (왼쪽→오른쪽)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    # 3. 모델 로드
    model = load_model(model_path)

    results = []
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        digit_img = thresh[y:y+h, x:x+w]
        # MNIST 스타일 전처리 (정사각형 패딩, 리사이즈, 정규화)
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
        digit_img = cv2.resize(padded, (28, 28))
        digit_img = digit_img.astype('float32') / 255.0
        digit_img = digit_img.reshape(1, 28, 28, 1)

        # 예측
        prediction = model.predict(digit_img)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit]
        results.append((predicted_digit, confidence))

        # 각 숫자별 결과 파일 저장
        output_path = os.path.join(output_dir, f"{student_id}_3_{idx+1}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"예측된 숫자: {predicted_digit}\n")
            f.write(f"신뢰도: {confidence:.4f}\n")
            f.write("\n각 숫자별 확률:\n")
            for i, prob in enumerate(prediction[0]):
                f.write(f"숫자 {i}: {prob:.4f}\n")

    print(f"총 {len(results)}개의 숫자를 예측하였습니다.")

# main 부분: 1.png, 2.png, 6.png, 9.png 예측
if __name__ == "__main__":
    student_id = "2025254011"
    # 예측에 사용할 모델 파일명을 2025254011.h5로 고정
    model_path = os.path.join(os.path.dirname(__file__), "2025254011.h5")
    image_list = ["1.png", "2.png", "6.png", "9.png"]
    for image_name in image_list:
        image_path = os.path.join(os.path.dirname(__file__), image_name)
        output_path = os.path.join(os.path.dirname(__file__), f"{student_id}_{image_name.split('.')[0]}.txt")
        predict_digit(model_path, image_path, output_path) 