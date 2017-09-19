import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from numpy import array
# 手动生成一个随机的平面点分布，并画出来
np.random.seed(0)
#X, y = make_moons(2000,noise=0.2)
#print(X)
#print(y)
X=array([[0,1],[2,3],[4,4],[2,0],[5,2],[6,3]])
y=array([1,1,1,0,0,0])
#print(X)
#print(y)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()


def plot_decision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
from sklearn.neighbors import KNeighborsClassifier
    # 咱们先来瞄一眼逻辑斯特回归对于它的分类效果
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)


    # 画一下决策边界
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()