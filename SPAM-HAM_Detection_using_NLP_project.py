#!/usr/bin/env python
# coding: utf-8

# # SPAM/HAM Classification Using NLP

# In[6]:


import nltk
import pandas as pd
import numpy as np

get_ipython().system('pip install nltk')


# In[36]:


dataset=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\SpamCollectionDataset.csv")
dataset.head()


# In[70]:


dataset['Message'][0]


# In[71]:


dataset['Message'][4]


# Shape of the Dataset

# In[72]:


print("Input dataset has {} many rows and {} many columns".format(len(dataset),len(dataset.columns)))


# How many Spam and Ham are there

# In[8]:


print("Out of {} rows, {} many rows are spam and {} many rows are ham".format(len(dataset),
                                                                             len(dataset[dataset['Category']=='spam']),
                                                                             len(dataset[dataset['Category']=='ham'])))


# How much missing data is there?

# In[74]:


print("Number of null in Category:{}".format(dataset['Category'].isnull().sum()))


# In[75]:


print("Number of null in Message:{}".format(dataset['Message'].isnull().sum()))


# Pre-Processing the data- Removing Punctuation

# In[37]:


import string

def discard_punct(text):
    text_nopunct="".join([char for char in text if char not in string.punctuation])
    return text_nopunct

dataset['Clean_Message']=dataset['Message'].apply(lambda x: discard_punct(x))
dataset.head()


# Tokeniztaion

# In[38]:


import re
def tokenize(text):
    tokens=re.split('\W',text)
    return tokens

dataset['tokenized_Message']=dataset['Message'].apply(lambda x:tokenize(x.lower()))
dataset.head()


# Remove Stopwords

# In[39]:


stopwords=nltk.corpus.stopwords.words('english')

stopwords[0:10]


# In[41]:


def discard_stopwords(tokenized_list):
    text=[word for word in tokenized_list if word not in stopwords]
    return text

dataset['nostop_Message']=dataset['tokenized_Message'].apply(lambda x:discard_stopwords(x))
dataset.head()


# Lemmatization

# In[15]:


wn=nltk.WordNetLemmatizer()

def lemmitizing(tokenized_list):
    text=[wn.lemmatize(word) for word in tokenized_list]
    return text

dataset['lemmatized_Message']=dataset['tokenized_Message'].apply(lambda x:lemmitizing(x))
dataset.head()


# Vectorization and Count Vectorization

# In[42]:


from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W',text)
    text=[wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text

count_vect=CountVectorizer(analyzer=clean_text)
x_counts=count_vect.fit_transform(dataset['Message'])
print(x_counts.shape)
    


# Apply Count Vectorizer to a small sample

# In[43]:


data_sample=dataset[0:20]

count_vect_sample=CountVectorizer(analyzer=clean_text)
x_counts_sample=count_vect_sample.fit_transform(data_sample['Message'])
print(x_counts_sample.shape)


# Sparse Matrix

# In[44]:


x_counts_sample


# In[46]:


x_counts_df=pd.DataFrame(x_counts_sample.toarray())
x_counts_df


# TF-IDF

# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect=TfidfVectorizer(analyzer=clean_text)
X_tfidf=tfidf_vect.fit_transform(dataset['Message'])

print(X_tfidf.shape)


# Apply TfidfVectorizer to a small sample

# In[48]:


data_sample=dataset[0:20]

tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample['Message'])

print(X_tfidf_sample.shape)


# In[21]:


X_tfidf_df= pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_df.columns= tfidf_vect_sample.get_feature_names()
X_tfidf_df.head()


# Feature Engineering- Feature Creation

# In[25]:


dataset=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\SpamCollectionDataset.csv")
dataset.head()


# Create Feature for text message length

# In[26]:


dataset['Message_Length']=dataset['Message'].apply(lambda x:len(x)-x.count(" "))

dataset.head()


# Create feature for % of the text that is punctuation

# In[27]:


def count_punct(text):
    count= sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)- text.count(" ")),3)*100

dataset['punct%']=dataset['Message'].apply(lambda x:count_punct(x))
dataset.head()


# In[28]:


import matplotlib.pyplot as plt
import numpy as np

bins=np.linspace(0,200,40)

plt.hist(dataset['Message_Length'],bins)
plt.title('Message length Distribution')
plt.show()


# In[101]:


bins=np.linspace(0,50,40)

plt.hist(dataset['punct%'],bins)
plt.title('Punctuation length Distribution')
plt.show()


# # Building Machine Learning Classifiers using Random Forest Model
# 

# In[2]:


import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


# In[51]:


dataset=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\SpamCollectionDataset.csv")

dataset.head()


# In[4]:


def count_punct(text):
    count= sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)- text.count(" ")),3)*100

dataset['punct%']=dataset['Message'].apply(lambda x:count_punct(x))
dataset.head()


# In[23]:


def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W',text)
    text=[wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect=TfidfVectorizer(analyzer=clean_text)
X_tfidf=tfidf_vect.fit_transform(dataset['Message'])


# In[31]:


X_features=pd.concat([dataset['Message_Length'],dataset['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# # Model using K-Fold cross Validation

# In[63]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

rf=RandomForestClassifier(n_jobs=1)
k_fold=KFold(n_splits=5)
X_features.columns = X_features.columns.astype(str)

cross_val_score(rf, X_features, dataset['Category'], cv=k_fold, scoring="accuracy", n_jobs=1)


# # Model using Train Test Split

# In[65]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[66]:


X_train,X_test,y_train, y_test=train_test_split(X_features,dataset['Category'],test_size=0.3)


# In[67]:


rf=RandomForestClassifier(n_estimators=500,max_depth=20,n_jobs=-1)
rf_model=rf.fit(X_train, y_train)


# In[68]:


sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[0:10]


# In[69]:


y_pred=rf_model.predict(X_test)

precision, recall, fscore, support=score(y_test, y_pred, pos_label='spam', average='binary')


# In[70]:


print('Precision {} / Recall {} / Accuracy {}'.format(round(precision,3),
                                                     round(recall,3),
                                                     round((y_pred==y_test).sum()/len(y_pred),3)))


# In[ ]:




