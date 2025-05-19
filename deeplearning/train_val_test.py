# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 검증 데이터 분리
validation_size = 1000
x_train, t_train = x_train[:-validation_size], t_train[:-validation_size]
x_val, t_val = x_train[-validation_size:], t_train[-validation_size:]

# 1. 실험용 설정==========
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                          output_size=10, use_dropout=True, dropout_ratio=0.15)

optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
val_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

# 2. 훈련 시작==========
for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0:
        # 훈련 데이터의 정확도
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        
        # 검증 데이터의 정확도
        val_acc = network.accuracy(x_val, t_val)
        val_acc_list.append(val_acc)
        
        # 테스트 데이터의 정확도
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        
        # 손실값
        train_loss = network.loss(x_train, t_train)
        train_loss_list.append(train_loss)
        
        print("epoch:" + str(epoch_cnt) + " | " + 
              "train acc:" + str(train_acc) + " | " + 
              "val acc:" + str(val_acc) + " | " +
              "test acc:" + str(test_acc))
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 3. 그래프 그리기==========
markers = {'train': 'o', 'val': 's', 'test': 'D'}
x = np.arange(max_epochs)

# 정확도 그래프
plt.subplot(2, 1, 1)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, val_acc_list, marker='s', label='validation', markevery=10)
plt.plot(x, test_acc_list, marker='D', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("훈련/검증/테스트 데이터에 대한 정확도")

# 손실값 그래프
plt.subplot(2, 1, 2)
plt.plot(x, train_loss_list, marker='o', label='train', markevery=10)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 3.0)
plt.legend(loc='upper right')
plt.title("훈련 데이터에 대한 손실값")

plt.tight_layout()
plt.show()

# 4. 최적의 에폭 찾기==========
best_epoch = np.argmax(val_acc_list)
print("최적의 에폭:", best_epoch)
print("검증 데이터 최고 정확도:", val_acc_list[best_epoch])
print("테스트 데이터 정확도:", test_acc_list[best_epoch]) 