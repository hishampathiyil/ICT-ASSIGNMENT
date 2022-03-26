#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # 1.Reading dataset into python environment

# In[2]:


data=pd.read_csv(r"C:\Users\hisham\Desktop\New folder\train_file.csv")
data.head()


# # 2.Making Id as index

# In[6]:


data.set_index('ID').head()


# # 3.Details of Dataset

# In[ ]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isna().sum()


# In[9]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),vmin=-0.5,vmax=0.5,annot=True,linewidth=0.4)


# # 4.Filling missing values present in columns

# In[28]:


y=data['MaterialType']


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 42)
x_train.shape


# In[30]:


data1=data
data1['MaterialType']=data1['MaterialType'].fillna(data1['MaterialType'].mean())
data1.isna().sum()


# In[13]:


data1['Publisher']=data1['Publisher'].fillna(data1['Publisher'].mode()[0])
data1['MaterialType']=data1['MaterialType'].fillna(data1['MaterialType'].mode()[0])
data1.isna().sum()


# # 5.Checking and handling outliers

# In[15]:


Q1=data['Checkouts'].quantile(0.25)
Q3=data['Checkouts'].quantile(0.75)
IQR=Q3-Q1
upperlimit=Q1+1.5*IQR
lowerlimit=Q3-1.5*IQR


# In[16]:


data[data['Checkouts']>upperlimit]
data[data['Checkouts']<lowerlimit]


# In[17]:


newdata=data[data['Checkouts']<upperlimit]
newdata.shape


# In[18]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.boxplot(data['Checkouts'])
plt.subplot(2,2,2)
sns.boxplot(newdata['Checkouts'])


# In[19]:


print(data['CheckoutMonth'].skew())


# In[20]:


data['CheckoutMonth'].describe()


# In[26]:


data.to_csv(r"C:\Users\hisham\Desktop\New folder\File_Name.csv", index = False)


# In[ ]:




