import numpy as np
import matplotlib.pyplot as plt

def lin_reg(X,Y,a,itr):
    m = X.shape[0]
    w,b = 0,0

    for i in range(itr):
        Y_pred = w * X + b
        dw = (1/m) * np.sum((Y_pred - Y) * X)
        db = (1/m) * np.sum((Y_pred - Y))
        w = w - a * dw
        b = b - a * db
       
    return (w,b)


train_data = np.loadtxt("train.csv",skiprows=1,delimiter=',')
X_train = train_data[:,0]
Y_train = train_data[:,1]

w,b = lin_reg(X_train,Y_train,0.00001,10000)

test_data = np.loadtxt("test.csv",skiprows=1,delimiter=',')
X_test = test_data[:,0]
Y_test = test_data[:,1]

c = X_test.shape[0]
Y_test_pred = w * X_test + b

rmse = (np.sum((Y_test_pred - Y_test) ** 2)/c)**0.5
print(rmse)

plt.scatter(X_train,Y_train,s=10,label='Train Values')
plt.scatter(X_test,Y_test_pred,s=10,c='red',label = 'Predicted Values')
plt.legend()
plt.show()





