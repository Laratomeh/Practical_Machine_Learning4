#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import gensim
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn import metrics
import nltk
import re
import time
import contractions
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, make_scorer


# In[2]:


# Reading files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


# Using part of the data due to the very large size,
# and the inability to use it completely due to the small size of the memory. 
train = train.sample(frac = 0.1,  random_state=42)
train.info()


# In[4]:


# Rename the 'Class Index' columns
train = train.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)
test = test.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)


# In[5]:


# Define a function for text cleaning
def clean(text):
    # convert text to lowercase
    text = text.lower()  
    # git red of non word carachters
    text = re.sub(r'\W', ' ', text)
    # remove digits
    text = re.sub(r'\d', ' ', text)
    # remove single carachters
    text = re.sub(r'\s+[a-z]\s+', ' ', text, flags=re.I)
    # remove single carachters at the start of the sentence 
    text = re.sub(r'^[a-z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'^\s', '', text) # space at beggining
    text = re.sub(r'\s$', '', text) # space at ending
    # Removing contractions "abbreviations"
    text = contractions.fix(text)
    # get rid of stopwords
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    # # Stemming words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # Lemmatizing words
    # lemmatizer = WordNetLemmatizer()
    # text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


# In[6]:


# Apply function to clean the text
train["tokenized"] = train["Description"].apply(clean)
test["tokenized"] = test["Description"].apply(clean)


# In[7]:


# Train Doc2Vec model
documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(train['tokenized'])]
model = gensim.models.doc2vec.Doc2Vec(documents, workers=3, alpha=0.05, vector_size=80, min_count=30, min_alpha=0.005)
model.train(documents, total_examples=model.corpus_count, epochs=20, start_alpha=0.03, end_alpha=-0.012)
x_train = model.dv.vectors
x_test = []
for doc in range(len(test)):
    vector = model.infer_vector(test['tokenized'][doc].split())
    x_test.append(vector)


# In[8]:


# Train and test labels
Train_labels = train.Class_Index
Test_labels = test.Class_Index


# In[9]:


# Creat model object
mp = make_pipeline(
    TruncatedSVD(),
    OPTICS())

# Get the parameters of the model
mp.get_params().keys()


# In[18]:


# Define the parameter grid
param_grid = {
    'truncatedsvd__n_components': [50, 100],
    'optics__p': [1.5, 2],
    'optics__algorithm': ['auto', 'ball_tree'],
    'optics__leaf_size':[30,40],
    'optics__eps':[0.3, 0.6, 0.9],
    'optics__metric':['cityblock','cosine', 'euclidean','l1','l2','manhattan','minkowski'],
    'optics__min_cluster_size':[10,100,500]
}


# In[19]:


# Create an instance of GridSearchCV
clf = GridSearchCV(mp, param_grid=param_grid, cv=2, scoring=adjusted_mutual_info_score, verbose = 5)

# Fit the GridSearchCV instance to the text data
clustering = clf.fit(x_train)


# In[20]:


# Predicting test labels
pred_labels = OPTICS(metric='minkowski').fit_predict(x_test) 

# Compute the metrics
rand_score = rand_score(Test_labels, pred_labels)
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)
silhouette_score = silhouette_score(x_test, Test_labels, metric='euclidean')

print('Rand score:',rand_score)
print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)
print('Silhouette score:',silhouette_score)


# In[23]:


pred_labels = OPTICS(metric='l1').fit_predict(x_test)      

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[25]:


pred_labels = OPTICS(metric='l2').fit_predict(x_test)  

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[26]:


pred_labels = OPTICS(metric='cityblock').fit_predict(x_test)  

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[27]:


pred_labels = OPTICS(metric='cosine').fit_predict(x_test)  

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[28]:


pred_labels = OPTICS(metric='euclidean').fit_predict(x_test)  

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[29]:


pred_labels = OPTICS(metric='manhattan',min_cluster_size=500).fit_predict(x_test)  

# Compute the metrics
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)

print('Adjusted rand:',adjusted_rand)
print('Adjusted mutual:',adjusted_mutual)
print('Normalized mutual:',normalized_mutual)


# In[ ]:




