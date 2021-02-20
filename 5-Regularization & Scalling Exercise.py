#%%
# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# TODO: Load the data,Assign the data to predictor and outcome variables
train_data = pd.read_csv('data/5-Regularization & Scalling Exercise.csv', header=None)
        #因原data没有header，则载入csv的时候如果不加header=None就会让第一行data被当做header
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

X.describe() #来看看原来数据各feature的统计学特征




#%%
#这一cell是用来scaling features的，对比做与不做scaling的差别:
# TODO: Create the standardization scaling object.
scaler = StandardScaler()
# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

df = pd.DataFrame(X_scaled) #将np.array变成pd.DataFrame才能用describe()
df.describe() #来看看原来数据scale之后的各feature的统计学特征




#%%
# TODO: Create the linear regression model with lasso regularization.
lasso_reg_noScl = Lasso()
lasso_reg_noScl.fit(X, y) #Fit the model without scaling.

lasso_reg_Scl = Lasso() 
lasso_reg_Scl.fit(X_scaled, y) #Fit the model with scaling.

#用普通的Linear Regression来看一下原来LR本身的coefficients都是多少，和Lasso Regularization
#之后的coefficients对比一下
lr_noScl = LinearRegression()
lr_noScl.fit(X, y)

lr_Scl = LinearRegression()
lr_Scl.fit(X_scaled, y)

# TODO: Retrieve the coefficients from the regression model.
reg_coef_noScl = ' / '.join(lasso_reg_noScl.coef_.round(2).astype(str))
reg_coef_Scl = ' / '.join(lasso_reg_Scl.coef_.round(2).astype(str))
lr_coef_noScl = ' / '.join(lr_noScl.coef_.round(2).astype(str))
lr_coef_Scl = ' / '.join(lr_Scl.coef_.round(2).astype(str))

print('[无scaling的Lasso]: ', reg_coef_noScl, 
      '\n', '-'*20,'\n', 
      '[有scaling的Lasso]: ', reg_coef_Scl,
      '\n', '-'*20,'\n', 
      '[无scaling的LR]: ', lr_coef_noScl,
      '\n', '-'*20,'\n', 
      '[有scaling的LR]: ', lr_coef_Scl)

