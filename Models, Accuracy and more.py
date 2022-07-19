#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("F:\St.clair\Sem-3\excel files\\hotaldataClean1.csv")


# In[4]:


df.head()


# In[7]:


df.tail()


# ### Data Types

# In[6]:


df.dtypes


# ##### Changing the data types

# In[37]:


df["IsCanceled"] = df["IsCanceled"].astype("category")
df["Meal"] = df["Meal"].astype("category")
df["MarketSegment"] = df["MarketSegment"].astype("category")
df["IsRepeatedGuest"] = df["IsRepeatedGuest"].astype("category")
df["ArrivalDateMonth"] = df["ArrivalDateMonth"].astype("category")
df["Children"] = df["Children"].astype("category")
df["DistributionChannel"] = df["DistributionChannel"].astype("category")
df["Country"] = df["Country"].astype("category")
df["ReservedRoomType"] = df["ReservedRoomType"].astype("category")
df["AssignedRoomType"] = df["AssignedRoomType"].astype("category")
df["DepositType"] = df["DepositType"].astype("category")
df["Agent"] = df["Agent"].astype("category")
df["CustomerType"] = df["CustomerType"].astype("category")
df["ADR"] = df["ADR"].astype("str")
df["ReservationStatus"] = df["ReservationStatus"].astype("category")
df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])
df["Hotal"] = df["Hotal"].astype("category")
df["Children"] = df["Children"].astype("int")


# In[38]:


df.info()


# #### Rounding the values

# In[39]:


df.describe().round(0)


# #### Shape of the DataFrame

# In[40]:


df.shape


# ### List

# In[42]:


df.loc[df["ArrivalDateWeekNumber"] == 14,["ArrivalDateWeekNumber","LeadTime"]]


# In[190]:


df.loc[df["ArrivalDateMonth"] == "July",["ArrivalDateMonth","StaysInWeekendNights"]]


# ### Null values

# In[44]:


df.isnull().sum().sum()


# ### Duplicated values

# In[45]:


df.duplicated().sum()


# In[47]:


# Dropping duplicates
df.drop_duplicates(inplace = True)
df.duplicated().sum()


# # Outliers

# In[48]:


# libraries
import seaborn as sb
import matplotlib.pyplot as plt


# In[49]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['LeadTime'], bins = 10, ax=axes[0]) # indexing starts from 0 and we want histogram as a 1st figure thats why we are putting ax=axes[0]
plt.boxplot(df['LeadTime'], patch_artist = True)

axes[0].set_title('Distribution of the Leadtime of customers')
axes[1].set_title('Rnage of the date of arrival of customers')
plt.tight_layout()
plt.show()


# ### Removing outliers

# In[51]:


import numpy as np


# In[52]:


# Medain of the LeadTime
median = np.median(df["LeadTime"])
median


# In[53]:


# calculating Q1 for leadtime attribute
Q1_LeadTime = df["LeadTime"].quantile(0.25)
print("Q1_LeadTime:", int(Q1_LeadTime),"days")


# In[54]:


#calculating Q3 for leadtime attribute
Q3_LeadTime = df.LeadTime.quantile(0.75)
print("Q3_LeadTime:", int(Q3_LeadTime),"days")


# In[55]:


IQR_LeadTime = Q3_LeadTime - Q1_LeadTime
print("IQR_LeadTime", int(IQR_LeadTime),"days")


# In[56]:


uperBound_LeadTime = Q3_LeadTime + (1.5 * IQR_LeadTime)
uperBound_LeadTime


# In[57]:


df.loc[(df["LeadTime"]> uperBound_LeadTime),].shape[0]


# In[58]:


# making index
index = df.loc[(df["LeadTime"] > uperBound_LeadTime),].index
index


# In[59]:


# Dropping index
df = df.drop(index)


# In[60]:


df.shape[0]


# ### Checking outliers for each column

# In[63]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['ArrivalDateWeekNumber'], bins = 10, ax=axes[0])
plt.boxplot(df['ArrivalDateWeekNumber'], patch_artist = True)

axes[0].set_title('Distribution of the ArrivalDateWeekNumber')
axes[1].set_title('Range of the date of ArrivalDateWeekNumber')
plt.tight_layout()
# plt.savefig("hist of ArrivalDateDayOfMonth.png")
plt.show()


# In[64]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['ArrivalDateDayOfMonth'], bins = 10, ax=axes[0])
plt.boxplot(df['ArrivalDateDayOfMonth'], patch_artist = True)

axes[0].set_title('Distribution of the ArrivalDateDayOfMonth')
axes[1].set_title('Range of the date of ArrivalDateDayOfMonth')
plt.tight_layout()
# plt.savefig("hist of ArrivalDateDayOfMonth.png")
plt.show()


# In[65]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['StaysInWeekendNights'], bins = 10, ax=axes[0])
plt.boxplot(df['StaysInWeekendNights'], patch_artist = True)

axes[0].set_title('Distribution of the StaysInWeekendNights')
axes[1].set_title('Range of the date of StaysInWeekendNights')
plt.tight_layout()
# plt.savefig("hist of ArrivalDateDayOfMonth.png")
plt.show()


# In[66]:


Q1_StaysInWeekendNights = df["StaysInWeekendNights"].quantile(0.25)
print("Q1_StaysInWeekendNights:", int(Q1_StaysInWeekendNights),"days")


# In[67]:


Q3_StaysInWeekendNights= df.StaysInWeekendNights.quantile(0.75)
print("Q3_StaysInWeekendNights:", int(Q3_StaysInWeekendNights),"days")


# In[68]:


IQR_StaysInWeekendNights = Q3_StaysInWeekendNights - Q1_StaysInWeekendNights
print("IQR_StaysInWeekendNights", int(IQR_StaysInWeekendNights),"days")


# In[69]:


uperBound_StaysInWeekendNights = Q3_StaysInWeekendNights + (1.5 * IQR_StaysInWeekendNights)
uperBound_StaysInWeekendNights


# In[70]:


df.loc[df['StaysInWeekendNights'] > uperBound_StaysInWeekendNights, "StaysInWeekendNights"] = Q3_StaysInWeekendNights


# ### Outliers Removed

# In[71]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['StaysInWeekendNights'], bins = 10, ax=axes[0])
plt.boxplot(df['StaysInWeekendNights'], patch_artist = True)

axes[0].set_title('Distribution of the StaysInWeekendNights')
axes[1].set_title('Range of the date of StaysInWeekendNights')
plt.tight_layout()
# plt.savefig("hist of ArrivalDateDayOfMonth.png")
plt.show()


# In[73]:


fig, axes = plt.subplots(1,2, figsize=(20,4))
sb.histplot(df['StaysInWeekendNights'], bins = 10, ax=axes[0])
plt.boxplot(df['StaysInWeekendNights'], patch_artist = True)

axes[0].set_title('Distribution of the StaysInWeekendNights')
axes[1].set_title('Range of the date of StaysInWeekendNights')
plt.tight_layout()
# plt.savefig("hist of ArrivalDateDayOfMonth.png")
plt.show()


# # Visualizations

# In[74]:


from pandas.api import types


# In[76]:


# for loop
numeric = []
category = []
for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric.append(col)
    else: 
        category.append(col)
print("category:", category)


# In[77]:


df["IsCanceled"].unique()


# In[78]:


countChurned = df.IsCanceled[df['IsCanceled']==1].count()
countRetained = df.IsCanceled[df['IsCanceled']==0].count()

labels = ['Canceled', 'notCanceled']
slices = [countChurned, countRetained]
explode = [0.1,0]

plt.pie(slices, labels = labels,shadow=True,
       startangle = (+60), explode = explode, autopct = '%1.1f%%',
       wedgeprops = {'edgecolor' : 'black'})
plt.axis('equal')
plt.title("Churn pie distribution")
plt.legend()
plt.tight_layout()
plt.show()


# ### Saving the visualization

# In[81]:


plt.savefig('Churn pie distribution.jpg')


# In[82]:


df["IsRepeatedGuest"].unique()


# In[83]:


countChurned = df.IsRepeatedGuest[df['IsRepeatedGuest']==1].count()
countRetained = df.IsRepeatedGuest[df['IsRepeatedGuest']==0].count()

labels = ['not repeated', 'repeated']
slices = [countChurned, countRetained]
explode = [0.4,0]

plt.pie(slices, labels = labels,shadow=True,
       startangle = (+60), explode = explode, autopct = '%1.1f%%',
       wedgeprops = {'edgecolor' : 'black'})
plt.axis('equal')
plt.title("Repeated guest")
plt.legend()
plt.tight_layout()
# plt.savefig('Churn pie distribution.jpg')
plt.show()


# ### Question - what percentage of hotel reservations are cancelled?

# In[84]:


sb.countplot(x='Hotal',hue = 'IsCanceled',data=df)
plt.show()


# # Variance

# In[88]:


variance = df.var()
variance


# ### deleting attributes with nearly zero variance from the data set because they dont provide any information about the data set.

# In[90]:


variance.to_csv("variance.csv",index = True)


# In[91]:


df = df.drop(["ArrivalDateYear","StaysInWeekendNights","Adults","Children","Babies","PreviousCancellations","BookingChanges","RequiredCarParkingSpaces","TotalOfSpecialRequests"], axis = 1)


# In[93]:


variance = df.var()
variance


# # Pearson Correlation - It measures linear correlation

# In[95]:


corr = df.corr().round(2)
corr


# In[99]:


plt.figure(figsize=(20,20))
sb.heatmap(df.corr(), annot = True) # annot is for showing values
plt.title("Correlation matrix using pearson method")
plt.show()


# # Spearman - It compares the rank of data 

# ### for spearman we have to put method = spearman otherwise it takes pearson as default method

# In[101]:


corr1 = df.corr(method="spearman").round(2)
corr1


# In[102]:


plt.figure(figsize=(20,20))
sb.heatmap(df.corr(method="spearman"), annot = True)
plt.title("Spearman")
plt.show()


# In[113]:


df.columns


# # Normalization

# In[150]:


category


# In[151]:


# Because we dont want categorical data
df.columns[~df.columns.isin(category)]


# In[153]:


# input variable - which are not vategorical
# dfx = input variable
dfx = df[df.columns[~df.columns.isin(category)]]
dfx


# In[154]:


# convert the target variable into numeric
df['IsCanceled'] = df['IsCanceled'].astype(int)


# ### target variabel is IsCancelled because we want to know is reservation cancled or not 

# In[162]:


df1 = df.corr()['IsCanceled'].abs().sort_values(ascending = False)
df1


# ### MinMaxScaler

# In[158]:


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[159]:


scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(dfx),columns=dfx.columns)


# In[160]:


scaled_df


# In[163]:


dfy = pd.Series(df["IsCanceled"])
dfy.value_counts()


# # train_test split

# In[165]:


from sklearn.model_selection import train_test_split


# In[166]:


x_train, x_test, y_train, y_test = train_test_split(scaled_df,dfy,test_size = 0.3,random_state = 4)


# ### Data Balancing using SMOTE

# In[168]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()


# In[169]:


x_train, y_train = oversample.fit_resample(x_train, y_train)
y_train.value_counts()


# # KNN

# - the knn algorithm can be used for both classification and regression
# - but it is more commanly used for classification
# - the k nearest neighbour algorithm is based on the supervised learning technique
# - the knn method thinks that the new case/data and existing cases are comparable,
# - and it places the new case in the category that is closest to the existing categories.

# In[172]:


from sklearn.neighbors import KNeighborsClassifier


# In[173]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics


# In[183]:


knn = KNeighborsClassifier()
model = knn.fit(x_train,y_train) # fitting the knn model to training data
prediction_k = model.predict(x_test) # Predicting the test set
knn_acc = accuracy_score(y_test, prediction_k)*100 # Getting the accuracy score
print("accuracy of KNN :", round(knn_acc, 3),"%") # Here 3 means number of digits after dot


# # Logistic Regression

# In[180]:


from sklearn.linear_model import LogisticRegression


# In[182]:


Lr = LogisticRegression()
model = Lr.fit(x_train, y_train)
prediction_logistic = model.predict(x_test)
cf_metrix_logistic = confusion_matrix(y_test, prediction_logistic)
logistic_acc = accuracy_score(y_test, prediction_logistic)*100
print("accuracy of logistic regression :", round(logistic_acc,2),"%")


# ### Confusion metrics

# - Confusion metrics gives tabular form for the number of correct and incorrect predictions made by classifiers

# In[184]:


cf_metrix_logistic


# # Random Forest 

# In[187]:


from sklearn.ensemble import RandomForestClassifier


# In[189]:


RFC = RandomForestClassifier(n_estimators = 30) # Where, n_estimators represents the number of trees it will create
model = RFC.fit(x_train, y_train)
RFC_predicted = model.predict(x_test)
RFC_acc = accuracy_score(y_test, RFC_predicted)*100
print("accuracy of random forest :", round(RFC_acc,2),"%")

