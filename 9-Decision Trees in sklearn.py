#%%
# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data/9-Decision Trees in sklearn.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X_train = data[0:81,0:2]
y_train = data[0:81,2]

X_test = data[81:,0:2]  ##### Uda直接用training data来做prediction了，然后出图！
y_test= data[81:,2]

X_test.shape


#%%
# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)

# TODO: Fit the model.
model.fit(X_train, y_train)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X_test)
print(y_pred)


#%%
# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y_test, y_pred)
print(acc)
# %%
