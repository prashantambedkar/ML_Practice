#!/usr/bin/env python
# coding: utf-8

# ## Point 1

# In[26]:


# import necessary packages, and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[27]:


# dataframe creation with the help of pandas
df = pd.read_csv("task1.csv")
df.head()


# In[28]:


df.info() # informations of the attributes of the dataframe


# In[29]:


df.corr() # analysis of the correlation values


# In[30]:


corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns) # heat map using correlation method


# In[31]:


X = df.drop("y", axis=1) # X variable with first five columns
Y = df["y"] # Y variable with respect to y column


# In[32]:


# test and train data spliting
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state= 42)


# In[33]:


# regression model creation
reg_mod = LinearRegression()
model_val = reg_mod.fit(x_train,y_train)


# In[34]:


#RMSE value calcuation for train data
rmse_train = np.sqrt(mean_squared_error(y_train, model_val.predict(x_train)))
rmse_train


# In[35]:


# RMSE value calculation for test data
rmse_test = np.sqrt(mean_squared_error(y_test, model_val.predict(x_test)))
rmse_test


# In[36]:


# score of the model
model_val.score(x_train, y_train) 


# In[37]:


# cross validfation score of the model
cross_val_score(model_val, x_train,  y_train, cv= 10, scoring="r2").mean()


# In[38]:


y_predicted = reg_mod.predict(x_test) # prediction of the model
r2_score(y_test, y_predicted) # R2 score calcuation with respect to test, and predicted values of the model


# ## Point 2 

# In[39]:


df_new = df.drop(["x2","x3","x4","x5"], axis=1) # creation of new dataframe using two attributes, x1, and y
df_new.head()


# In[40]:


X1 = df_new.drop("y", axis=1) # define X1 by using x1 column value
Y1 = df_new["y"] # define Y1 by using y column value


# In[41]:


plt.scatter(X1, Y1, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x1')
plt.ylabel('y')
plt.show() # scatter plot between X1, and Y1


# In[42]:


df_new1 = df.drop(["x1","x2","x3","x4"], axis=1) # creation of new dataframe using two attributes, x5, and y
df_new1.head()


# In[43]:


X2 = df_new1.drop("y", axis=1) # define X2 by using x5 column value
Y2 = df_new1["y"] # define Y2 by using y column value


# In[44]:


plt.scatter(X2, Y2, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x5')
plt.ylabel('y')
plt.show() # scatter plot between X1, and Y1

