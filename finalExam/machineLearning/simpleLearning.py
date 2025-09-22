# 1. 예시 데이터
x = 2.0     # 입력값
y = 8.0     # 실제값

# 2. 초기값 설정
w = 0.0     # 가중치 초기값
b = 0.0     # 편향 초기값
lr = 0.01   # 학습률(learning rate)
epochs = 100  # 반복 횟수

# 3. 경사하강법 반복
for i in range(epochs):
    y_pred = w * x + b                  # 예측값
    loss = 0.5 * (y - y_pred)**2        # 손실함수

    grad_w = -(y - y_pred) * x          # w에 대한 편미분
    grad_b = -(y - y_pred)              # b에 대한 편미분

    # 4. 매개변수 업데이트
    w = w - lr * grad_w
    b = b - lr * grad_b

    # 5. 10회마다 중간 결과 출력
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

print(f"최종 가중치 w: {w:.4f}, 편향 b: {b:.4f}")
