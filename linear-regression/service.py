#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

#X,y = datasets.make_regression(n_samples=100, n_features=1, noise =20)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#visualize = plt.figure()
#plt.scatter(X, y)
#plt.show()

#importing dataset
dataset = pd.read_csv("fishermen_mercury.csv")

#storing features and actual results in numpy arrays 
X= dataset.iloc[:, :-2].values
Y_methyl_mercury=dataset.iloc[:, 7].values
Y_total_mercury=dataset.iloc[:, 7].values

#applying one-hot encoding for fishermen and fishpart columns
columnTransformer = ColumnTransformer([('encoder', 
										OneHotEncoder(), 
										[0])], 
									remainder='passthrough') 

X = np.array(columnTransformer.fit_transform(X), dtype = np.str) 

columnTransformer = ColumnTransformer([('encoder', 
										OneHotEncoder(), 
										[6])], 
									remainder='passthrough') 

X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

#print(X)