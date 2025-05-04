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

gamma = np.ones((1, hidden1_dim))
beta = np.zeros((1, hidden1_dim))
running_mean = np.zeros((1, hidden1_dim))
running_var = np.ones((1, hidden1_dim))
epsilon = 1e-7
gamma2 = np.ones((1, hidden2_dim))
beta2 = np.zeros((1, hidden2_dim))
running_mean2 = np.zeros((1, hidden2_dim))
running_var2 = np.ones((1, hidden2_dim))

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
    global running_mean, running_var, running_mean2, running_var2
    z1 = X @ W1 + b1
    batch_mean = np.mean(z1, axis=0, keepdims=True)
    batch_var = np.var(z1, axis=0, keepdims=True)
    
    running_mean = 0.9 * running_mean + 0.1 * batch_mean
    running_var = 0.9 * running_var + 0.1 * batch_var   
    z1_norm = (z1 - batch_mean) / np.sqrt(batch_var + epsilon)
    bn_output = gamma * z1_norm + beta
    a1= relu(bn_output)
    
    z2 = a1 @ W2 + b2
    batch_mean2 = np.mean(z2, axis=0, keepdims=True)
    batch_var2 = np.var(z2, axis=0, keepdims=True)
    
    running_mean2 = 0.9 * running_mean2+ 0.1 * batch_mean2
    running_var2 = 0.9 * running_var2 + 0.1 * batch_var2   
    z2_norm = (z2 - batch_mean2) / np.sqrt(batch_var2 + epsilon)
    bn_output2 = gamma2 * z2_norm + beta2
    
    a2 = relu(bn_output2)
    
    z3 = a2 @ W3 + b3
    t = softmax(z3)
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
        batch_mean = np.mean(z1, axis=0, keepdims=True)
        batch_var = np.var(z1, axis=0, keepdims=True)
        
        running_mean = 0.9 * running_mean + 0.1 * batch_mean
        running_var = 0.9 * running_var + 0.1 * batch_var   
        z1_norm = (z1 - batch_mean) / np.sqrt(batch_var + epsilon)
        bn_output = gamma * z1_norm + beta
        
        a1= relu(bn_output)
        
        z2 = a1 @ W2 + b2
        batch_mean2 = np.mean(z2, axis=0, keepdims=True)
        batch_var2 = np.var(z2, axis=0, keepdims=True)
        
        running_mean2 = 0.9 * running_mean2+ 0.1 * batch_mean2
        running_var2 = 0.9 * running_var2 + 0.1 * batch_var2   
        z2_norm = (z2 - batch_mean2) / np.sqrt(batch_var2 + epsilon)
        bn_output2 = gamma2 * z2_norm + beta2
        
        a2 = relu(bn_output2)
        
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy(t, y_batch)
        #################################################
        dz3 = (t - y_batch) / batch_size
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ W3.T
        dz2 = da2 * relu_derivative(bn_output2)
        
        dgamma2 = np.sum(dz2 * z2_norm, axis=0, keepdims=True)
        dbeta2 = np.sum(dz2, axis=0, keepdims=True)
        
        dz2_norm = dz2 * gamma2
        
        dbatch_var2 = np.sum(dz2_norm * (z2 - batch_mean2) * -0.5 * (batch_var2 + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean2 = np.sum(dz2_norm * -1 / np.sqrt(batch_var2 + epsilon), axis=0, keepdims=True) + dbatch_var2 * np.mean(-2 * (z2 - batch_mean2), axis=0, keepdims=True)
        
        dz2 = dz2_norm / np.sqrt(batch_var2 + epsilon) + dbatch_var2 * 2 * (z2 - batch_mean2) / batch_size + dbatch_mean2 / batch_size
        
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(bn_output)
        
        dgamma = np.sum(dz1 * z1_norm, axis=0, keepdims=True)
        dbeta = np.sum(dz1, axis=0, keepdims=True)
        
        dz1_norm = dz1 * gamma
        
        dbatch_var = np.sum(dz1_norm * (z1 - batch_mean) * -0.5 * (batch_var + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean = np.sum(dz1_norm * -1 / np.sqrt(batch_var + epsilon), axis=0, keepdims=True) + dbatch_var * np.mean(-2 * (z1 - batch_mean), axis=0, keepdims=True)
        
        dz1 = dz1_norm / np.sqrt(batch_var + epsilon) + dbatch_var * 2 * (z1 - batch_mean) / batch_size + dbatch_mean / batch_size
        
        dW1 = X_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2 
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
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
    