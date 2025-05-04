import numpy as np

def gradient_descent(f,x):
    h = 1e-4
    x=np.array(x)
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val =x[idx]
        
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val -h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

def calc(f, init_x, lr, step_num):
    x = init_x
    x_history = []
    
    for i in range(step_num):
        x_history.append(x.copy())
        
        grad = gradient_descent(f,x)
        x -= lr*grad
        
    return x

def function_2(x):
    if x.ndim ==1:
        return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)
    
init_x = [3.0, 4.0]
ans = calc(function_2, init_x, 0.1, 100)
print(ans)