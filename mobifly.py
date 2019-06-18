#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_iris


# In[4]:


data = pd.read_csv('/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv')
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.read_csv('/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv', names=names, skiprows=1)


# In[5]:


dataset.head()


# In[6]:


dataset


# In[7]:


iris_dataset=dataset.drop(['species'],axis=1)


# In[8]:


iris_dataset


# In[8]:


iris_dataset.isnull().any().any()


# # BOX PLOT 

# In[9]:


iris_dataset.plot(kind='box', subplots=True,layout=(2,2),sharex=False, sharey=False)
plt.show()


# In[31]:


iris_dataset['sepal-width'].plot(kind='box')
plt.show()


# # HISTOGRAM

# In[10]:


iris_dataset.hist()
plt.show()


# In[11]:


iris_dataset=dataset.drop(['species'],axis=1)


# In[89]:


iris_dataset


# # SCATTER MATRIX

# In[12]:


#just to see corelation between variables.
from pandas.plotting import scatter_matrix
scatter_matrix(iris_dataset)
plt.show()


# # SCATTER PLOT

# In[13]:


iris_dataset['sepal-length'].plot(style=".")


# In[14]:


iris_dataset['sepal-width'].plot(style=".")


# In[15]:


iris_dataset['petal-length'].plot(style=".")


# In[16]:


iris_dataset['petal-width'].plot(style=".")


# In[24]:


def draw_scatter(dataSet, field):
    dataSet[field].plot(style=".")
    


# In[25]:


draw_scatter(iris_dataset,'petal-width')


# # DENSITY PLOT

# In[20]:


iris_dataset.plot(kind='density',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[ ]:




