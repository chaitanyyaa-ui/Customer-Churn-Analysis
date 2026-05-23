#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Problem Statement  : You are the Data Scientist at a telecom company “Neo” whose customers are churning out to its competitors. You have to analyse the data of your company and find insights and stop your customers from churning out to other telecom companies.


# In[2]:


# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('F:\Dataset\customer_churn.csv')
df


# In[4]:


# Data Manupulation 
# Extract the 5th the column & store it in ‘customer_5’
customer_5 = df.iloc[:,4]
customer_5


# In[5]:


#Extract the 15th column & store it in ‘customer_15’
customer_15 = df['Contract']
customer_15


# In[6]:


# Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’
senior_male_electronic = df[(df['SeniorCitizen']==1 ) & (df['gender'] == 'Male') & (df['PaymentMethod']== 'Electronic check')]
senior_male_electronic


# In[7]:


# Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’
customer_total_tenure = df[(df['tenure']>70) | (df['MonthlyCharges']>100)]
customer_total_tenure


# In[8]:


#Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
two_mail_yes = df[(df['Contract'] == 'Two year') & (df['PaymentMethod'] == 'Mailed check') & (df['Churn'] == 'Yes')]
two_mail_yes


# In[9]:


#Extract 333 random records from the customer_churn dataframe & store the result in‘customer_333’
customer_333 = df.sample(333,random_state =42)
customer_333


# In[10]:


#Get the count of different levels from the ‘Churn’ column
df['Churn'].value_counts()


# In[11]:


#DATA VISUALIZATION
#a. Build a bar-plot for the ’InternetService’ column:
#i. Set x-axis label to ‘Categories of Internet Service’
#ii. Set y-axis label to ‘Count of Categories’
#iii. Set the title of plot to be ‘Distribution of Internet Service’
#iv. Set the color of the bars to be ‘orange’


# In[12]:


IS_Count  = df['InternetService'].value_counts()
IS_Count


# In[13]:


plt.bar(IS_Count.index,IS_Count.values, color = 'orange', label='Internet Services')
plt.xlabel('Categories of Internet Service')
plt.legend('Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()


# In[14]:


#b. Build a histogram for the ‘tenure’ column:
#i. Set the number of bins to be 30
#ii. Set the color of the bins to be ‘green’
#iii. Assign the title ‘Distribution of tenure’

#c. Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:
#i. Assign the points a color of ‘brown’
#ii. Set the x-axis label to ‘Tenure of customer’
#iii. Set the y-axis label to ‘Monthly Charges of customer’
#iv. Set the title to ‘Tenure vs Monthly Charges’

#d. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on the x-axis.

#b 
sns.histplot(x=df['tenure'],color = 'green',bins=30,hue=df['Churn'])
plt.title('‘Distribution of tenure’')
plt.show()





# In[15]:


#C 
plt.scatter(df['MonthlyCharges'],df['tenure'],color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')
plt.show()


# In[16]:


sns.boxplot(x='Contract', y='tenure',data=df)
plt.xlabel('Contract')
plt.ylabel('Tenure')
plt.show()


# In[17]:


#LINEAR REGRESSION 
#a. Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’
#i. Divide the dataset into train and test sets in 70:30 ratio.
#ii. Build the model on train set and predict the values on test set
#iii. After predicting the values, find the root mean square error
#iv. Find out the error in prediction & store the result in ‘error’
#v. Find the root mean square error
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


# In[18]:


x = df[['tenure']]
y=df['MonthlyCharges']


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.30, random_state = 42)


# In[20]:


lr = LinearRegression()


# In[21]:


#Train the model
lr.fit(x_train,y_train)


# In[22]:


#Test the model
lr_pred = lr.predict(x_test)


# In[23]:


rmse = np.sqrt(mean_squared_error(y_test,lr_pred))
rmse


# In[24]:


error = lr_pred-y_test
error


# In[25]:


r2_score(y_test,lr_pred)


# In[26]:


#a. Build a simple logistic regression model where dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’
#i. Divide the dataset in 65:35 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and get the accuracy score


# In[27]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

x = df[['MonthlyCharges']]
y = df['Churn']


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5, random_state = 42)


# In[29]:


log.fit(x_train,y_train)


# In[63]:


y_pred = log.predict(x_test)


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


confusion_matrix(y_test,y_pred)


# In[33]:


recall_score(y_test,y_pred,pos_label='Yes')


# In[ ]:


'''b) Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’
i. Divide the dataset in 80:20 ratio
ii. Build the model on train set and predict the values on test set
iii. Build the confusion matrix and get the accuracy score'''


# In[82]:


x = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


# In[83]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 42)


# In[84]:


log = LogisticRegression()


# In[85]:


log.fit(x_train, y_train)


# In[86]:


y_pred = log.predict(x_test)


# In[87]:


confusion_matrix(y_test, y_pred)


# In[88]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:


# Decision Tree:
#a. Build a decision tree model where dependent variable is ‘Churn’ & independent variable is ‘tenure’
#i. Divide the dataset in 80:20 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy


# In[57]:


x = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


# In[64]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# In[65]:


from sklearn.tree import DecisionTreeClassifier


# In[66]:


dt = DecisionTreeClassifier()


# In[67]:


dt.fit(x_train,y_train)


# In[68]:


y_pred = dt.predict(x_test)


# In[69]:


accuracy_score(y_test,y_pred)


# In[ ]:


#Random Forest:
#a. Build a Random Forest model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ and ‘MonthlyCharges’
#i. Divide the dataset in 70:30 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy


# In[70]:


x = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# In[73]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


rf= RandomForestClassifier(max_depth=5,n_estimators=100)


# In[79]:


rf.fit(x_train,y_train)


# In[80]:


y_pred = rf.predict(x_test)


# In[81]:


accuracy_score(y_test,y_pred)

