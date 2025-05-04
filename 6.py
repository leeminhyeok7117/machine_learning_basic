import numpy as np
import sys, os
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

input_dim = 784
hidden_dim = 50
output_dim = 10
learning_rate = 0.1
epochs = 10
batch_size = 100

np.random.seed(0)
W1 = np.random.randn(input_dim, hidden_dim) *0.01
W2 = np.random.randn(hidden_dim, output_dim) *0.01
b1 = np.zeros((1, hidden_dim))
b2 = np.zeros((1, output_dim))

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
    t = softmax(z2)
    y_pred = np.argmax(t, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true)
    
train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

loss_list = []
acc_list = []
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
        t = softmax(z2)
        loss = cross_entropy(t, y_batch)
        
        dL_dz2 = (t - y_batch) /  batch_size
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = dL_dz2 @ W2.T  
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2 
        

    epoch_list.append(epoch)
    loss_list.append(loss)
    acc = accuracy(x_test, t_test)
    acc_list.append(acc)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label='Accuracy', color='orange')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
    