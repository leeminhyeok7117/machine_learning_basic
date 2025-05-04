import numpy as np

w1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
w2 = np.array([[0.1,0.3],[0.2,0.4],[0.3,0.5]])
w3 = np.array([[0.1, 0.2], [0.3,0.4]])

b1 = np.array([0.3, 0.2, 0.1])
b2 = np.array([0.1, 0.2])
b3 = np.array([0.5, 0.1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def cal(x,w,b):
    c = np.dot(x,w)
    return c+b

X = np.array([1.0, 0.5])

result1 = cal(X,w1,b1)
result2 = sigmoid(result1)
result3 = cal(result2,w2,b2)
result4 = sigmoid(result3)
result5 = cal(result4,w3,b3)

print(result5)

result6 = cal(X,w1,b1)
result7 = relu(result6)
result8 = cal(result7,w2,b2)
result9 = relu(result8)
result10 = cal(result9,w3,b3)

print(result10)
