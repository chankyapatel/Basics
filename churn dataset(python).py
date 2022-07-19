#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("C:\\Users\\Patel Chankya\\Desktop\\Churn_Modelling.csv")


# In[4]:


df.head()


# In[6]:


df.head(10)


# In[7]:


df.tail()


# In[8]:


df.tail(10)


# In[9]:


df.head(5).T


# In[10]:


df.tail(5).T


# In[12]:


df.head(5).T.to_csv("head.csv")


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.describe().round(3)


# In[17]:


des1 = df.describe().round(0).T
des1.to_csv("des_trans.csv")


# In[18]:


df.keys()


# In[19]:


df.rename(columns = {'Geography' : 'Country'}, inplace = True)


# In[20]:


df.columns


# In[22]:


df.isnull()


# In[23]:


df.isnull().sum()


# In[24]:


df.isnull().sum().sum()


# In[25]:


(df.var(),0)


# In[26]:


df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)


# In[27]:


df["Gender"].unique()


# In[29]:


df["Country"].unique()


# In[30]:


# checking data type for all the variables
df.dtypes


# In[34]:


df["CreditScore"].dtypes


# In[38]:


df["Country"].dtypes


# In[43]:


df.nunique().sum()


# In[44]:


df["NumOfProducts"].unique()


# In[45]:


df["NumOfProducts"].value_counts()


# In[46]:


df["Gender"].value_counts()


# In[47]:


df["Exited"].dtypes


# In[55]:


import matplotlib.pyplot as plt


# In[87]:


# Plotting pie chart for percentage of exited customers

countChurned = df.Exited[df['Exited']==1].count()
countRetained = df.Exited[df['Exited']==0].count()

labels = ['Churn', 'Retained']
slices = [countChurned, countRetained]
explode = [0.5,0]

plt.pie(slices, labels = labels,shadow=True,
       startangle = (+60), explode = explode, autopct = '%1.5f%%',
       wedgeprops = {'edgecolor' : 'black'})
plt.axis('equal')
plt.title("Churn pie distribution")
plt.legend()
plt.tight_layout()
plt.savefig('Churn pie distribution.jpg')
plt.show()


# In[88]:


countChurned = df.Country[df['Country']==1].count()
countChurned


# In[119]:


France= df.Country[df['Country']=="France"].count()
Spain = df.Country[df['Country']=="Spain"].count()
Germany = df.Country[df['Country']=="Germany"].count()

labels = ['France', 'Spain', 'Gemany']
slices = [France, Spain, Germany]
explode = [0.1,0.1,0.1]

plt.pie(slices, labels = labels,shadow=True,
       startangle = (+90), explode = explode, autopct = '%1.2f%%',
       wedgeprops = {'edgecolor' : 'black'})
plt.axis('equal')
plt.title("Pie distribution of country")
plt.legend()
plt.tight_layout()
plt.savefig("Pie distribution of country")
plt.show()


# In[164]:


One = df.NumOfProducts[df['NumOfProducts']==1].count()
Two = df.NumOfProducts[df['NumOfProducts']==2].count()
Three = df.NumOfProducts[df['NumOfProducts']==3].count()
Four = df.NumOfProducts[df['NumOfProducts']==4].count()

labels = [1, 2, 3, 4]
values = [One, Two, Three, Four]
colors = ['green','red', 'yellow', 'orange']

plt.bar(labels, values, color = colors,
        width = 0.5)

plt.title("Num Of Products Distribution")
plt.xlabel("products")
plt.ylabel("Num of Products")
plt.legend(values[0:3])
plt.show()


# In[154]:


import seaborn as sns


# In[161]:


# visualizing distribution of country and exited attributes
sns.countplot (x='Country', hue = 'Exited', data = df)
plt.savefig('Country vs Exited.jpg')


# In[165]:


df["Tenure"].unique()


# In[172]:


One = df.Tenure[df['Tenure']==2].count()
Two = df.Tenure[df['Tenure']==1].count()
Three = df.Tenure[df['Tenure']==8].count()
Four = df.Tenure[df['Tenure']==7].count()
Five = df.Tenure[df['Tenure']==4].count()
Six = df.Tenure[df['Tenure']==6].count()
Seven = df.Tenure[df['Tenure']==3].count()
Eight = df.Tenure[df['Tenure']==10].count()
Nine = df.Tenure[df['Tenure']==5].count()
Ten = df.Tenure[df['Tenure']==9].count()
Eleven = df.Tenure[df['Tenure']==0].count()

labels = [2,  1,  8,  7,  4,  6,  3, 10,  5,  9,  0]
values = [One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Eleven]
colors = ['green']

plt.bar(labels, values, color = colors,
        width = 0.5)

plt.title("Tenure")
plt.show()


# In[173]:


sns.countplot (x='Gender', hue = 'Exited', data = df)
plt.savefig('Gender vs Exited.jpg')


# In[175]:


sns.countplot (x='HasCrCard', hue = 'Exited', data = df)
plt.savefig('HasCrCard vs Exited.jpg')


# In[186]:


fig, axes = plt.subplots(1,2, figsize=(20,6))
sns.histplot(df['Age'],bins=10,ax=axes[0])
plt.boxplot(df['Age'],patch_artist=True)

axes[0].set_title('Distribution of the Age of customers')
axes[1].set_title('Range of thr date of Age of customers')
plt.tight_layout()  #plt.savefig("hist of ArrivalDateDayofmonth.png")
plt.show()


# In[ ]:




