#%%
# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#%%
# TODO: Assign the data to predictor and outcome variables
train_data = pd.read_csv('data/4-Polynomial Regression Exercise.csv')
train_data = train_data.sort_values(by=['Var_X']) #这一句不加出来的图也是乱的
y = train_data['Var_Y'].values.reshape(-1, 1)
X = np.array(train_data['Var_X'].values)
X = np.reshape(X,(20,1)) 
    #将X变2D array还可用：X2 = train_data['Var_X'].values.reshape(-1, 1)

#%%
# TODO: Create a PolynomialFeatures object, then fit and transform the predictor feature
poly_feat = PolynomialFeatures(degree = 4) #可用下图看出degree越高越overfit
X_poly = poly_feat.fit_transform(X)

#%%
# TODO: Create a LinearRegression object and fit it to the polynomial predictor features
# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!
polyLR_model = LinearRegression(fit_intercept = False)
polyLR_model.fit(X_poly, y)
y_pred = polyLR_model.predict(X_poly)

plt.scatter(X, y);
plt.plot(X_poly[:,1],y_pred,color='g')
    #因为X_poly已经变形为2D了，如果后面如果不加[:,1]出来的图就是乱的

# %%
