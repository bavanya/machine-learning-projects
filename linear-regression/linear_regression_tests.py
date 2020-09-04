import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=1, noise =20, random_state =4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

from linear_regression import LinearRegression

regressor = LinearRegression(0.001,10)
regressor.fit(X_train,y_train)
predicted = regressor.predict(X_test)

def mean_square_error(y_actual, y_predicted):
    return np.mean((y_actual-y_predicted)**2)

error = mean_square_error(y_test,predicted)
print(error)