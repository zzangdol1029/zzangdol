import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# w1, w2 범위 설정
w1 = np.linspace(-5, 15, 100)
w2 = np.linspace(-5, 15, 100)
W1, W2 = np.meshgrid(w1, w2)

# 손실 함수 정의
Z = 21 * (10 - (W1 + W2 + 1))**2

# 3D 그래프 그리기
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('3D Loss Surface for L(w1, w2) = 21(10 - (w1 + w2 + 1))^2')

plt.show()
