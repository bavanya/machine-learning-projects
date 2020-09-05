#importing libraries
#from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA  
import model

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

########################################################################################################################################
#split data using train_test_split function
X_train_conv, X_test_conv, y_train_conv, y_test_conv = train_test_split( X, y_methyl_mercury, test_size=0.2, random_state=0)
X_train_conv_1, X_test_conv_1, y_train_conv_1, y_test_conv_1 = train_test_split( X, y_total_mercury, test_size=0.2, random_state=0)

#train_pct_index = int(0.8*len(X))  
#X_train_conv, X_test_conv = X[:train_pct_index], X[train_pct_index:]
#y_train_conv, y_test_conv = y_methyl_mercury[:train_pct_index], y_methyl_mercury[train_pct_index:]

################################################################
#call the regressor model and find error and store it
regressor = model.linear_regression(0.001, 10)
regressor.fit(X_train_conv,y_train_conv)
predicted = regressor.predict(X_test_conv)

regressor_1 = model.linear_regression(0.001, 10)
regressor_1.fit(X_train_conv_1,y_train_conv_1)
predicted_1 = regressor_1.predict(X_test_conv_1)
#################################################################

#calling using sklearn
regressor = LinearRegression()
X_train_conv = X_train_conv.astype(np.float64)
y_train_conv = y_train_conv.astype(np.float64)
regressor.fit(X_train_conv,y_train_conv)
X_test_conv = X_test_conv.astype(np.float64)
y_pred_0= regressor.predict(X_test_conv)
print(y_pred_0)

#applying metrics on the model
mae = metrics.mean_absolute_error(y_test_conv, y_pred_0)
mse = metrics.mean_squared_error(y_test_conv, y_pred_0)
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
rms = np.sqrt(mse)
print('Root Mean Squared Error:', rms)

#mae_scratch = np.abs(np.subtract(y_test_conv,y_pred_0)).mean()
#mse_scratch = np.square(np.subtract(y_test_conv, y_pred_0)).mean()
#rms_scratch = np.sqrt(mse_scratch)

#calling using sklearn
regressor_1 = LinearRegression()
X_train_conv_1 = X_train_conv_1.astype(np.float64)
y_train_conv_1 = y_train_conv_1.astype(np.float64)
regressor_1.fit(X_train_conv_1,y_train_conv_1)
X_test_conv_1 = X_test_conv_1.astype(np.float64)
y_pred_0_1= regressor.predict(X_test_conv_1)
print(y_pred_0_1)
    
#apply metrics on the model
mae_ = metrics.mean_absolute_error(y_test_conv_1, y_pred_0_1)
mse = metrics.mean_squared_error(y_test_conv_1, y_pred_0_1)
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
rms = np.sqrt(mse)
print('Root Mean Squared Error:', rms)

#################################################################

########################################################################################################################################
#spliting datset with kfold validation

kfold = KFold(n_splits=4)

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_methyl_mercury[train_index], y_methyl_mercury[test_index]

    ################################################################
    #call the regressor model from another file and find error and append to list
    regressor = model.linear_regression(0.001, 10)
    regressor.fit(X_train,y_train)
    predicted = regressor.predict(X_test)
    #################################################################

    #calling using sklearn
    regressor = LinearRegression()
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    regressor.fit(X_train,y_train)
    X_test = X_test.astype(np.float64)
    y_pred_1= regressor.predict(X_test)
    print(y_pred_1)
        
    #apply metrics on the model
    mae_ = metrics.mean_absolute_error(y_test, y_pred_1)
    mse = metrics.mean_squared_error(y_test, y_pred_1)
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    rms = np.sqrt(mse)
    print('Root Mean Squared Error:', rms)

    #################################################################

kfold_1 = KFold(n_splits=4)

for train_index, test_index in kfold_1.split(X):
    X_train_1, X_test_1 = X[train_index], X[test_index]
    y_train_1, y_test_1 = y_total_mercury[train_index], y_total_mercury[test_index]

    ################################################################
    #call the regressor model from another file and find error and append to list
    regressor_1 = model.linear_regression(0.001, 10)
    regressor_1.fit(X_train_1,y_train_1)
    predicted_1 = regressor_1.predict(X_test_1)
    #################################################################

    #calling using sklearn
    regressor_1 = LinearRegression()
    X_train_1 = X_train_1.astype(np.float64)
    y_train_1 = y_train_1.astype(np.float64)
    regressor_1.fit(X_train_1,y_train_1)
    X_test_1 = X_test_1.astype(np.float64)
    y_pred_1_1= regressor.predict(X_test_1)
    print(y_pred_1_1)

    #apply metrics on the model
    mae_ = metrics.mean_absolute_error(y_test_1, y_pred_1_1)
    mse = metrics.mean_squared_error(y_test_1, y_pred_1_1)
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    rms = np.sqrt(mse)
    print('Root Mean Squared Error:', rms)
    #################################################################

########################################################################################################################################
#splitting dataset with repeated k-fold

repeated_kfold = RepeatedKFold(n_splits=4, n_repeats=2)

for train_index, test_index in repeated_kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_methyl_mercury[train_index], y_methyl_mercury[test_index]

    ################################################################
    #call the regressor model from another file and find error and append to list
    regressor = model.linear_regression(0.001, 10)
    regressor.fit(X_train,y_train)
    predicted = regressor.predict(X_test)
    #################################################################

    #calling using sklearn
    regressor = LinearRegression()
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    regressor.fit(X_train,y_train)
    X_test = X_test.astype(np.float64)
    y_pred_1= regressor.predict(X_test)
    print(y_pred_1)

    #apply metrics on the model
    mae_ = metrics.mean_absolute_error(y_test, y_pred_1)
    mse = metrics.mean_squared_error(y_test, y_pred_1)
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    rms = np.sqrt(mse)
    print('Root Mean Squared Error:', rms)

    #################################################################

for train_index, test_index in repeated_kfold.split(X):
    X_train_1, X_test_1 = X[train_index], X[test_index]
    y_train_1, y_test_1 = y_total_mercury[train_index], y_total_mercury[test_index]

    ################################################################
    #call the regressor model from another file and find error and append to list
    regressor_1 = model.linear_regression(0.001, 10)
    regressor_1.fit(X_train_1,y_train_1)
    predicted_1 = regressor_1.predict(X_test_1)
    #################################################################

    #calling using sklearn
    regressor_1 = LinearRegression()
    X_train_1 = X_train_1.astype(np.float64)
    y_train_1 = y_train_1.astype(np.float64)
    regressor_1.fit(X_train_1,y_train_1)
    X_test_1 = X_test_1.astype(np.float64)
    y_pred_2_1= regressor.predict(X_test_1)
    print(y_pred_2_1)

    #apply metrics on the model
    mae_ = metrics.mean_absolute_error(y_test_1, y_pred_2_1)
    mse = metrics.mean_squared_error(y_test_1, y_pred_2_1)
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    rms = np.sqrt(mse)
    print('Root Mean Squared Error:', rms)
    
#########################################################################################################################################

########################################### other cross validation schemes that can be included are:#############################################
    #leave-one-out: 
    #leave-p-out
    #shuffle-and-split
    #stratified k-fold
    #stratified-shuffle-split

############## metrices that can be added for regression models are###############
    #R squared or coefficient of determination
    #adjusted R squared

########################################################################################################################################

