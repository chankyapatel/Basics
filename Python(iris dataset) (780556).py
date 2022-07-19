#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_data = load_iris()
x_iris = iris_data.data
y_iris = iris_data.target


# In[2]:


x_iris


# In[3]:


y_iris


# In[4]:


import numpy as np


# In[5]:


print("shape of x_iris data is {}".format(x_iris.shape))
print("shape of y_iris data is {}".format(y_iris.shape))


# In[6]:


y_iris = y_iris.reshape((150,))
print("Dimesnsions of y_iris are {}".format(y_iris.shape))
print(y_iris)


# In[7]:


y_iris = y_iris.T
print("Dimensions of y_iris are {}".format(y_iris.shape))
y_iris


# In[10]:


x_iris_train, x_iris_test, y_iris_train, y_iris_test = train_test_split(x_iris, y_iris, random_state = 0, shuffle = True)


# In[11]:


print("shape of x_iris_train data is {}".format(x_iris_train.shape))
print("shape of x_iris_test data is {}".format(x_iris_test.shape))
print("shape of y_iris_train data is {}".format(y_iris_train.shape))
print("shape of y_iris_test data is {}".format(y_iris_test.shape))
#74.6 training data


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


clf = KNeighborsClassifier(n_neighbors=1)


# In[17]:


clf.fit(x_iris_train, y_iris_train)


# In[18]:


acc_train = clf.score(x_iris_train, y_iris_train)
acc_test = clf.score(x_iris_test, y_iris_test)

print("training set accuracy: {:.2f}".format(acc_train))
print("test set accuracy: {:.2f}".format(acc_test))


# In[19]:


x_iris_new = [[5.2, 3.1, 1.1, 0.3]]


# In[23]:


n_neighbors = [1, 3, 5, 9, 11]
iris_train_accuracies = []
iris_test_accuracies = []
clf_new = []

for n in n_neighbors:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(x_iris_train, y_iris_train)
    acc_train = clf.score(x_iris_train, y_iris_train)
    acc_test = clf.score(x_iris_test, y_iris_test)
    clf_new.append(clf)
    iris_train_accuracies.append(acc_train)
    iris_test_accuracies.append(acc_test)
    
    print("training set accuracy: ",iris_train_accuracies)
    print("test set accuracy: ",iris_test_accuracies)


# In[24]:


import matplotlib.pyplot as plt

n_neighbors = [1, 3, 5, 9, 11]

plt.plot(n_neighbors, iris_train_accuracies, label = "train")
plt.plot(n_neighbors, iris_test_accuracies, label = "test")
plt.ylim(0, 1.1)
plt.xlabel("number of neighbors")
plt.ylabel("accuracy")
plt.title("k nearest neighbors/nNumber of Neighbors vs Accuracy for iris data")
plt.legend(loc=0)
plt.show()


# In[ ]:




