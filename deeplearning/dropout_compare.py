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

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 1. 실험용 설정==========
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                          output_size=10, use_dropout=True, dropout_ratio=0.15)
network_no_dropout = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                     output_size=10)

optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []
train_loss_list_no_dropout = []
train_acc_list_no_dropout = []
test_acc_list_no_dropout = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    grads_no_dropout = network_no_dropout.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    optimizer.update(network_no_dropout.params, grads_no_dropout)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        train_acc_no_dropout = network_no_dropout.accuracy(x_train, t_train)
        test_acc_no_dropout = network_no_dropout.accuracy(x_test, t_test)
        train_acc_list_no_dropout.append(train_acc_no_dropout)
        test_acc_list_no_dropout.append(test_acc_no_dropout)
        
        train_loss = network.loss(x_train, t_train)
        train_loss_no_dropout = network_no_dropout.loss(x_train, t_train)
        train_loss_list.append(train_loss)
        train_loss_list_no_dropout.append(train_loss_no_dropout)
        
        print("epoch:" + str(epoch_cnt) + " | " + 
              "train acc:" + str(train_acc) + " | " + 
              "test acc:" + str(test_acc) + " | " +
              "train acc(no dropout):" + str(train_acc_no_dropout) + " | " + 
              "test acc(no dropout):" + str(test_acc_no_dropout))
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 3. 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.subplot(2, 1, 1)
plt.plot(x, train_acc_list, marker='o', label='train (with dropout)', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test (with dropout)', markevery=10)
plt.plot(x, train_acc_list_no_dropout, marker='x', label='train (without dropout)', markevery=10)
plt.plot(x, test_acc_list_no_dropout, marker='D', label='test (without dropout)', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Dropout 적용 여부에 따른 정확도 비교")

plt.subplot(2, 1, 2)
plt.plot(x, train_loss_list, marker='o', label='train (with dropout)', markevery=10)
plt.plot(x, train_loss_list_no_dropout, marker='x', label='train (without dropout)', markevery=10)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 3.0)
plt.legend(loc='upper right')
plt.title("Dropout 적용 여부에 따른 손실값 비교")

plt.tight_layout()
plt.show() 