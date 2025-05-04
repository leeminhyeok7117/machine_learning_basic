# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)                       
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    
    def predict(self,x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y
    
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        
    def loss(self,x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient_mmm(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)
    
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)
            
            x[idx] = tmp_val - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            
            x[idx] = tmp_val
            it.iternext()
            
        return grad
    
    def numerical_gradient(self, x, t):     
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = self.numerical_gradient_mmm(loss_W, self.params['W1'])
        grads['b1'] = self.numerical_gradient_mmm(loss_W, self.params['b1'])
        grads['W2'] = self.numerical_gradient_mmm(loss_W, self.params['W2'])
        grads['b2'] = self.numerical_gradient_mmm(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        grads = {}
        batch_num = x.shape[0]
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        da1 = np.dot(dy, W2.T)  
        dz1 = da1 * z1 * (1 - z1)
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads
    
    
    
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.05

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iterations (x100)")
plt.ylabel("loss")
plt.ylim(0, 9)
plt.xlim(0,10000)
plt.show()


