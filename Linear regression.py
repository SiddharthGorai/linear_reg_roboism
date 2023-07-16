import numpy as np
import matplotlib.pyplot as plt

def lin_reg(X,Y,a,itr):
    m = X.shape[0]
    w,b = 0,0
    Y_pred = w * X + b
    # print(Y_pred)
    for i in range(itr):
        dw = (1/m) * np.sum((Y_pred - Y) * X)
        db = (1/m) * np.sum((Y_pred - Y))
        tmp_w = w - a * dw
        tmp_b = b - a * db
        w = tmp_w
        b = tmp_b
    return (w,b)


train_data = np.loadtxt("train.csv",skiprows=1,delimiter=',')
X = train_data[:,0]
Y = train_data[:,1]

print(lin_reg(X,Y,0.0001,10000))


