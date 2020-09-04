import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X,y = datasets.make_regression(n_samples=100, n_features=1, noise =20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

visualize = plt.figure()
plt.scatter(X, y)
plt.show()