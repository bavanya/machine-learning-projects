import numpy as np

class linear_regression:

    def __init__(self, learning_rate, maximum_iterations):
        self.learning_rate = learning_rate
        self.maximum_iterations = maximum_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #implement forward pass and back propagation
        samples_count, features_count = X.shape
        self.weights = np.random.rand(features_count)
        self.bias = 0

        for _ in range(self.maximum_iterations):
            y_predict = np.dot(X, self.weights) + self.bias

            dw = (1/samples_count)*np.dot(X.T, (y_predict-y))
            db = (1/samples_count)*np.sum(y_predict-y)
            self.weights -= self.weights*dw
            self.bias -= self.bias*db
        #pass

    def predict(self, X):
        pass
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    
