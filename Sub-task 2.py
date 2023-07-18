#!/usr/bin/env python
# coding: utf-8

# In[30]:


# import of necessary libraries, and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[31]:


df = pd.read_csv("task2.csv") # dataframe creation with the help of pandas
df.head()


# In[32]:


X = df.drop(['y'], axis=1) # X variable with first two columns
Y = df['y'] # Y variable with respect to y column


# In[33]:


# test and train data spliting
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.25, random_state=27)


# In[34]:


clf_model = MLPClassifier(hidden_layer_sizes=(15,), max_iter=500, alpha=5,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001) # model 1 creation with the help of MLPClassifier


# In[35]:


clf_model.fit(x_train, y_train) # model 1 fit with respect to train data
y_pred = clf_model.predict(x_test) # prediction of the trained model 1


# In[36]:


acc1 = accuracy_score(y_test, y_pred) # accuracy score for the model 1
acc1


# In[37]:


gen_error1 = 1 - acc1 # generalization error for the model 1
gen_error1


# In[38]:


rmse = np.sqrt(mean_squared_error(y_train, clf_model.predict(x_train))) # RMSE value calculation of the model 1 with respect to train data
rmse


# In[39]:


rmse1 = np.sqrt(mean_squared_error(y_test, clf_model.predict(x_test))) # RMSE value calculation of the model 1 with respect to test data
rmse1


# In[40]:


clf_model1 = MLPClassifier(hidden_layer_sizes=(15,), max_iter=500, alpha=20,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001) # model 2 creation with the help of MLPClassifier


# In[41]:


clf_model1.fit(x_train, y_train) # model 2 fit with respect to train data
y_pred1 = clf_model1.predict(x_test) # prediction of the trained model 2


# In[42]:


acc2 = accuracy_score(y_test, y_pred1) # accuracy score for the model 2
acc2


# In[43]:


gen_error2 = 1 - acc2 # generalization error for the model 2
gen_error2


# In[44]:


rmse3 = np.sqrt(mean_squared_error(y_train, clf_model1.predict(x_train))) # RMSE value calculation of the model 2 with respect to train data
rmse3


# In[45]:


rmse4 = np.sqrt(mean_squared_error(y_test, clf_model1.predict(x_test))) # RMSE value calculation of the model 2 with respect to test data
rmse4


# In[46]:


X_val = df.drop(['y'], axis=1) # X variable with first two columns
Y_val = df['y'] # Y variable with respect to y column


# In[47]:


mod1 = MLPClassifier(hidden_layer_sizes=(15), alpha=5, random_state=1)
mod1.fit(X_val, Y_val) # model 1 creation and fitting


# In[48]:


y1_pred = mod1.predict(X_val) # prediction value of model 1


# In[49]:


mod2 = MLPClassifier(hidden_layer_sizes=(15,), alpha=20, random_state=1)
mod2.fit(X_val, Y_val)  # model 2 creation and fitting


# In[50]:


y2_pred = mod2.predict(X_val) # prediction value of model 2


# In[51]:


fig, ax = plt.subplots(figsize=(5, 5)) # figure dimension setting
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y1_pred, cmap='viridis', label='Model 1 (Alpha = 5)') # plot for Model 1
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y2_pred, cmap='viridis', label='Model 2 (Alpha = 20)') # plot for Model 2

ax.set_xlabel('Model 1') # x label setting
ax.set_ylabel('Model 2') # y label setting

ax.legend() # add legend in the graph

plt.show() # plot of scatter plot

