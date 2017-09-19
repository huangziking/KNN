import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from numpy import array

# 手动生成一个随机的平面点分布，并画出来
#np.random.seed(0)
#X, y = make_moons(200, noise=0.20)
type_X=array([[0,2,4,2,5,6],[1,3,4,0,2,3]])
#type_Y=(1,3,4,0,2,3)
#type_X.append(0,2,4,2,5,6)
#type_Y.append(1,3,4,0,2,3)
#X=(0,2,4,2,5,6)
#Y=(1,3,4,0,2,3)
plt.scatter(type_X[:,0], type_X[:,1], s=40, c= type_X[:,1],cmap=plt.cm.Spectral)
#plt.scatter(X, Y, s=40, c=Y, cmap=plt.cm.Spectral)
plt.show()
def plot_decision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = type_X[:, 0].min() - .5, type_X[:, 0].max() + .5
    y_min, y_max = type_X[:, 1].min() - .5, type_X[:, 1].max() + .5
    #x_min, x_max = 0 - .5, 6 + .5
    #y_min, y_max = 0 - .5, 4 + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(type_X[:,0], type_X[:,1],  cmap=plt.cm.Spectral)
    from sklearn.neighbors import KNeighborsClassifier
    # 咱们先来瞄一眼逻辑斯特回归对于它的分类效果
    clf = KNeighborsClassifier()
    clf.fit(type_X,2)

    # 画一下决策边界
    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("KNN")
    plt.show()