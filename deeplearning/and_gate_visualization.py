import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

class ANDGate:
    def __init__(self):
        self.w = np.random.randn(2)  # 가중치 초기화
        self.b = np.random.randn()   # 편향 초기화
        self.lr = 0.2                # 학습률
        
    def forward(self, x):
        return sigmoid(np.dot(x, self.w) + self.b)
    
    def train(self, x, t):
        # 순전파
        y = self.forward(x)
        
        # 역전파
        dy = y - t
        dw = np.dot(x.T, dy)
        db = np.sum(dy)
        
        # 가중치 업데이트
        self.w -= self.lr * dw
        self.b -= self.lr * db
        
        return y

# 학습 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([0, 0, 0, 1])

# AND 게이트 인스턴스 생성
and_gate = ANDGate()

# 학습 과정 기록
w_history = []
b_history = []
loss_history = []

# 학습
epochs = 1000
for epoch in range(epochs):
    # 모든 학습 데이터에 대해 학습
    for i in range(len(X)):
        y = and_gate.train(X[i], T[i])
        loss = mean_squared_error(y, T[i])
        
        # 기록
        w_history.append(and_gate.w.copy())
        b_history.append(and_gate.b)
        loss_history.append(loss)

# 결과 시각화
plt.figure(figsize=(10, 8))

# w_history를 NumPy 배열로 변환
w_history = np.array(w_history)

# 하나의 그래프에 모든 변화를 표시
plt.plot(w_history[:, 0], label='Weight 1', color='blue')
plt.plot(w_history[:, 1], label='Weight 2', color='green')
plt.plot(b_history, label='Bias', color='red')
plt.plot(loss_history, label='Loss', color='purple')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('AND Gate Learning Process')
plt.legend()
plt.grid(True)
plt.show()

# 최종 학습 결과 출력
print("\n최종 학습 결과:")
print("가중치:", and_gate.w)
print("편향:", and_gate.b)
print("\n예측 결과:")
for i in range(len(X)):
    y = and_gate.forward(X[i])
    print(f"입력: {X[i]}, 예측값: {y:.4f}, 실제값: {T[i]}") 