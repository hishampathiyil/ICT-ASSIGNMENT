#!/usr/bin/env python
# coding: utf-8

# # Case Study on Probability for Data Science

# To make a suitable machine learning algorithm to predict if the mushroom is 
# edible or poisonous (e or p) using the given dataset.
# (Along with other ML algorithms, Naïve Bayes’ Classifier should be applied)

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df=pd.read_csv(r"C:\Users\hisham\Downloads\mushrooms.csv")
df.head()


# In[16]:


df.shape


# In[17]:


df.info()


# In[18]:


df.describe(include = "object")


# In[19]:


df.isna().sum()


# In[21]:


df.dtypes


# In[22]:


df.columns


# In[23]:


for column in df.columns:
    print(f"{column} -> {df[column].nunique()}, {df[column].unique()}")


# In[24]:


df['class'].value_counts()


# In[25]:


x = df.drop(['class'], axis =1)
y = df['class']


# In[26]:


X = df.drop(columns=['class', 'veil-type'], axis = 1)
y = df['class']


# In[28]:


onehot_columns = []
label_columns = []
for column in X.columns:
    label_columns.append(column) if X[column].nunique()>3 else onehot_columns.append(column)
print(f"Onehot columns = {onehot_columns}")
print(f"Label columns = {label_columns}")


# In[29]:


X = pd.get_dummies(data = X, columns=onehot_columns)


# In[30]:


from sklearn.preprocessing import LabelEncoder
for column in label_columns:
    X[column] = LabelEncoder().fit_transform(X[column])


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[44]:


from sklearn.metrics import classification_report, confusion_matrix
def check_model_metrices(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix = \n', confusion_matrix(y_test, y_pred))


# # Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(multi_class = 'multinomial')
logistic_model.fit(X_train,y_train)
y_pred= logistic_model.predict(X_test)


# In[47]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score
print(metrics.classification_report(y_test, y_pred))


# # Linear SVM

# In[48]:


from sklearn.svm import SVC
linear = SVC(kernel='linear')
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)
check_model_metrices(y_test, linear_pred)


# # KNN

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def generate_kNN_model(x_train, y_train, x_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    knn_model.fit(x_train, y_train)
    return knn_model.predict(x_test)

# Optimizing 'k' or 'n-neighbers' value
def find_optimal_k(x_train, y_train, x_test):
    accur_dict = dict()
    for k in np.arange(3,16):
        y_pred = generate_kNN_model(x_train, y_train, x_test, k)
        accur_dict[k] = accuracy_score(y_test, y_pred)
    #Plot
    plt.plot(list(accur_dict.keys()),list(accur_dict.values()), marker ='o')
    plt.title('k-Values vs Accuracy')
    plt.show()
    optimal_k = max(accur_dict, key = lambda x: accur_dict[x])
    print('Best k value = ', optimal_k)
    return optimal_k
k_optimal = find_optimal_k(X_train, y_train, X_test)
knn_y_pred = generate_kNN_model(X_train, y_train, X_test, k_optimal)
print(f'kNN classifier with k = {k_optimal} has :: \n')
check_model_metrices(y_test, knn_y_pred)


# # Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
check_model_metrices(y_test, linear_pred)


# # Naive Bayes Classifier

# In[57]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB().fit(X_train, y_train)
bnb_pred = bnb.predict(X_test)
check_model_metrices(y_test, bnb_pred)


# In[58]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
check_model_metrices(y_test, gnb_pred)


# in this comparison I found that KNN provides extreme accuracy
