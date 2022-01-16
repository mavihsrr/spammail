#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# data collection and pre-processing

# In[2]:


raw_mail_data = pd.read_csv(r"C:\Users\shiva\Desktop\SpamMail detector\mail_data.csv")

print(raw_mail_data)


# In[3]:


#replacing the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[4]:


raw_mail_data.head()


# In[5]:


#checking the size of our dataset
mail_data.shape


# LABEL ENCODING

# In[6]:


#labelling spam mail as 0 and ham mail as 1

mail_data.loc[mail_data['Category']== 'spam', 'Category',]=0
mail_data.loc[mail_data['Category']== 'ham', 'Category',]=1


# In[7]:


mail_data.head()


# seperating the data set as texts and label 

# In[8]:


X = mail_data["Message"]
Y = mail_data["Category"]


# In[9]:


print(X)


# In[10]:


print(Y)


# Splitting the data into train and test data

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)


# In[12]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[14]:


# transforming the text data to feature vectors that can be used as input to the logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#converting Y_train and Y_test as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype("int")


# In[15]:


print(X_train_features)


# training our logisticregression model

# In[16]:


model = LogisticRegression()


# In[17]:


model.fit(X_train_features,Y_train)


# evalulating our model

# In[21]:


#prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy1 = accuracy_score(Y_train,prediction_on_training_data )


# In[23]:


print("The accuracy on training data is - ",accuracy1 )


# In[24]:


#prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy2 = accuracy_score(Y_test,prediction_on_test_data )


# In[26]:


print("The accuracy on test data is - ",accuracy2 )


# building a predictive system

# In[27]:


input_mail = ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...']

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#predictions
prediction = model.predict(input_data_features)
print(prediction)

if prediction[0]==0:
    print("Spam mail")
    
else:
    print("Ham mail")


# In[ ]:




