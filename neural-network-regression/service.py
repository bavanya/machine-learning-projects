#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA  

#X,y = datasets.make_regression(n_samples=100, n_features=1, noise =20)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#visualize = plt.figure()
#plt.scatter(X, y)
#plt.show()

#importing dataset
dataset = pd.read_csv("fishermen_mercury.csv")

#storing features and actual results in numpy arrays 
X= dataset.iloc[:, :-2].values
y_methyl_mercury=dataset.iloc[:, 7].values
y_total_mercury=dataset.iloc[:, 7].values

#preprocessing
#the dataset has been provided with label encoding
#converting to make the features binary instead of ordinal.
#Also to avoid the curse of dimensionality, applied PCA after one hot encoding
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

#feature scaling
#standardization is performed by StandardScaler function
sc = StandardScaler() 
X = sc.fit_transform(X)

#applying principle component analysis
pca = PCA(n_components = 2) 
X = pca.fit_transform(X) 

