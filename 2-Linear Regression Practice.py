# 1. Load the data
# The data is in the file called "bmi_and_life_expectancy.csv".
# Use pandas read_csv to load the data into a dataframe (don't forget to import pandas!)
# Assign the dataframe to the variable bmi_life_data.
#%%
import pandas as pd
bmi_life_data = pd.read_csv('data/2-bmi_and_life_expectancy.csv',delimiter = ',')
x = bmi_life_data.loc[:][['BMI']]  #不加最靠近BMI这一层的[]下面的.fit方程会出错
y = bmi_life_data.loc[:][['Life expectancy']]  #选取单一col必须在前面加上[:]先选取所有行
    #或者写成bmi_life_data[['BMI']],  bmi_life_data[['Life expectancy']]


# 2. Build a linear regression model
# Create a regression model using scikit-learn's LinearRegression and assign it to bmi_life_model.
# Make and fit the linear regression model to the data.
#%%
from sklearn.linear_model import LinearRegression
bmi_life_model = LinearRegression()
bmi_life_model.fit(x, y)


# 3. Predict using the model
# Predict using a BMI of 21.07931 and assign it to the variable laos_life_exp.
#%%
laos_life_exp = bmi_life_model.predict([[21.07931]])  #上面如果用了直接[]选择col，这里就可以：.predict(21.07931)
laos_life_exp

#%% <<完全正确！>>