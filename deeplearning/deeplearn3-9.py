import numpy as np

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=np.int_)

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print("W1.shape : " , W1.shape)   # (2, 3)
print("X.shape : ", X.shape)    # (2,)
print("B1.shape : ", B1.shape)   # (3,)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print("A1 : ", A1)
print("Z1 : ", Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("Z1.shap : " , Z1.shape)
print("W2.shape : " ,W2.shape)
print("B2.shape : " , B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print("A2 : ", A2)
print("Z2 : ", Z2)

def indetyfy_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = indetyfy_function(A3)

print("A3 : ", A3)
print("Y : ", Y)

