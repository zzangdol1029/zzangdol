import numpy as np
import matplotlib.pyplot as plt


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=np.int_)


# x값의 범위 설정
x = np.arange(-5.0, 5.0, 0.1)

# 각 함수 계산
sigmoid_y = sigmoid(x)
step_y = step_function(x)

# 결과 시각화
plt.plot(x, sigmoid_y, label="Sigmoid")
plt.plot(x, step_y, label="Step Function", linestyle='--')
plt.title("Sigmoid vs Step Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
