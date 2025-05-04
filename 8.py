import numpy as np
import sys, os
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

input_dim = 784
hidden1_dim = 50
hidden2_dim = 50
output_dim = 10
learning_rate = 0.1
epochs = 10
batch_size = 100

np.random.seed(0)
W1 = np.random.randn(input_dim, hidden1_dim) *0.1
W2 = np.random.randn(hidden1_dim, hidden2_dim) *0.1
W3 = np.random.randn(hidden2_dim, output_dim) *0.1
b1 = np.zeros((1, hidden1_dim))
b2 = np.zeros((1, hidden2_dim))
b3 = np.zeros((1, output_dim))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
    
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(preds, targets):
    return -np.sum(targets * np.log(preds + 1e-9)) / preds.shape[0]

def accuracy(X, y_true):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    z3 = a2 @ W3 + b3
    t = softmax(z3)
    y_pred = np.argmax(t, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true)

def sgd(W, b, dW, db, learning_rate):
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def momentum(W, b, dW, db, vW, vb, learning_rate, alpha=0.9):
    vW = alpha * vW - learning_rate * dW
    vb = alpha * vb - learning_rate * db
    W += vW
    b += vb
    return W, b, vW, vb

def adagrad(W, b, dW, db, hW, hb, learning_rate, epsilon=1e-7):
    hW += dW **2
    hb += db **2
    W -= learning_rate * dW / (np.sqrt(hW) + epsilon)
    b -= learning_rate * db / (np.sqrt(hb) + epsilon)
    return W, b ,hW, hb

def adam_update(param, dparam, m, v, epsilon = 1e-7, beta1=0.9, beta2=0.999, t=1):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

loss_list_adam = []
loss_list_sgd = []
loss_list_momentum = []
loss_list_adagrad = []
acc_list_adam = []
acc_list_sgd = []
acc_list_momentum = []
acc_list_adagrad = []
epoch_list = []

for epoch in range(epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]
        
        z1 = X_batch @ W1 + b1
        a1= sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy(t, y_batch)
        
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        W1, b1 = sgd(W1, b1, dL_dW1, dL_db1, learning_rate)
        W2, b2 = sgd(W2, b2, dL_dW2, dL_db2, learning_rate)
        W3, b3 = sgd(W3, b3, dL_dW3, dL_db3, learning_rate)
        
    epoch_list.append(epoch)
    loss_list_sgd.append(loss)
    acc = accuracy(x_test, t_test)
    acc_list_sgd.append(acc)
    print(f"Epoch_sgd {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    epoch_list = []
    
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3)
vb3 = np.zeros_like(b3)
W1 = np.random.randn(input_dim, hidden1_dim) *0.1
W2 = np.random.randn(hidden1_dim, hidden2_dim) *0.1
W3 = np.random.randn(hidden2_dim, output_dim) *0.1
b1 = np.zeros((1, hidden1_dim))
b2 = np.zeros((1, hidden2_dim))
b3 = np.zeros((1, output_dim))

for epoch in range(epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]
        
        z1 = X_batch @ W1 + b1
        a1= sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy(t, y_batch)
        
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        W1, b1, vW1, vb1 = momentum(W1, b1, dL_dW1, dL_db1, vW1, vb1, learning_rate)
        W2, b2, vW2, vb2 = momentum(W2, b2, dL_dW2, dL_db2, vW2, vb2,  learning_rate)    
        W3, b3, vW3, vb3 = momentum(W3, b3, dL_dW3, dL_db3, vW3, vb3, learning_rate)     
        
    epoch_list.append(epoch)
    loss_list_momentum.append(loss)
    acc = accuracy(x_test, t_test)
    acc_list_momentum.append(acc)
    print(f"Epoch_momentum {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    epoch_list = []

hW1 = np.zeros_like(W1)
hb1 = np.zeros_like(b1)
hW2 = np.zeros_like(W2)
hb2 = np.zeros_like(b2)
hW3 = np.zeros_like(W3)
hb3 = np.zeros_like(b3)
W1 = np.random.randn(input_dim, hidden1_dim) *0.1
W2 = np.random.randn(hidden1_dim, hidden2_dim) *0.1
W3 = np.random.randn(hidden2_dim, output_dim) *0.1
b1 = np.zeros((1, hidden1_dim))
b2 = np.zeros((1, hidden2_dim))
b3 = np.zeros((1, output_dim))

for epoch in range(epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]
        
        z1 = X_batch @ W1 + b1
        a1= sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy(t, y_batch)
        
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        W1, b1, hW1, hb1 = adagrad(W1, b1, dL_dW1, dL_db1, hW1, hb1, learning_rate)
        W2, b2, hW2, hb2 = adagrad(W2, b2, dL_dW2, dL_db2, hW2, hb2,  learning_rate)    
        W3, b3, hW3, hb3 = adagrad(W3, b3, dL_dW3, dL_db3, hW3, hb3, learning_rate)   

        
    epoch_list.append(epoch)
    loss_list_adagrad.append(loss)
    acc = accuracy(x_test, t_test)
    acc_list_adagrad.append(acc)
    print(f"Epoch_adagrad {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    epoch_list = []
    
mW1 = np.zeros_like(W1)
mW2 = np.zeros_like(W2)
mW3 = np.zeros_like(W3) 
vW1 = np.zeros_like(W1)
vW2 = np.zeros_like(W2)
vW3 = np.zeros_like(W3)
W1 = np.random.randn(input_dim, hidden1_dim) *0.1
W2 = np.random.randn(hidden1_dim, hidden2_dim) *0.1
W3 = np.random.randn(hidden2_dim, output_dim) *0.1
b1 = np.zeros((1, hidden1_dim))
b2 = np.zeros((1, hidden2_dim))
b3 = np.zeros((1, output_dim))

for epoch in range(epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]
        
        z1 = X_batch @ W1 + b1
        a1= sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy(t, y_batch)
        
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        W1, mW1, vW1 = adam_update(W1, dL_dW1, mW1, vW1)
        W2, mW2, vW2 = adam_update(W2, dL_dW2, mW2, vW2)
        W3, mW3, vW3 = adam_update(W3, dL_dW3, mW3, vW3)
        
        
    epoch_list.append(epoch)
    loss_list_adam.append(loss)
    acc = accuracy(x_test, t_test)
    acc_list_adam.append(acc)
    print(f"Epoch_adam {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    epoch_list = []
    
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_list_sgd, label='SGD')
plt.plot(loss_list_momentum, label='Momentum')
plt.plot(loss_list_adagrad, label='Adagrad')
plt.plot(loss_list_adam, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list_sgd, label='SGD')
plt.plot(acc_list_momentum, label='Momentum')
plt.plot(acc_list_adagrad, label='Adagrad')
plt.plot(acc_list_adam, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
