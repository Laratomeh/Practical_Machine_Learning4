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
import contractions
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, make_scorer
from scipy.cluster.hierarchy import dendrogram, linkage


# In[32]:


# Reading files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[33]:


# Using part of the data due to the very large size,
# and the inability to use it completely due to the small size of the memory. 
train = train.sample(frac = 0.005,  random_state=42)
train.info()


# In[34]:


# Rename the 'Class Index' columns
train = train.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)
test = test.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)


# In[35]:


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


# In[36]:


# Apply function to clean the text
train["tokenized"] = train["Description"].apply(clean)
test["tokenized"] = test["Description"].apply(clean)


# In[37]:


# Train Doc2Vec model
documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(train['tokenized'])]
model = gensim.models.doc2vec.Doc2Vec(documents, workers=3, alpha=0.05, vector_size=80, min_count=30, min_alpha=0.005)
model.train(documents, total_examples=model.corpus_count, epochs=20, start_alpha=0.03, end_alpha=-0.012)
x_train = model.dv.vectors
x_test = []
for doc in range(len(test)):
    vector = model.infer_vector(test['tokenized'][doc].split())
    x_test.append(vector)


# In[38]:


# Train and test labels
Train_labels = train.Class_Index
Test_labels = test.Class_Index


# In[15]:


# Training the model
for x, y in zip(range(1, 11, 1), range(1, 11, 1)):
    optics = OPTICS(metric='l2', min_samples=y)
    optics.fit(x_train)
    preds = optics.fit_predict(x_test) 
    adjusted_rand = metrics.adjusted_rand_score(Test_labels, preds)
    adjusted_mutual = metrics.adjusted_mutual_info_score(Test_labels, preds)
    normalized_mutual = metrics.normalized_mutual_info_score(Test_labels, preds)
    # Print the values
    print(x,':', 'Min_samples:',y, 'Adjusted_Rand:','%.6f'% adjusted_rand,'Adjusted_Mutual:',
          '%.6f'% adjusted_mutual, 'Normalized_Mutual:','%.6f'% normalized_mutual)


# In[39]:


# Print out dendrogram which visualize the data stucture and cluster numbers
dendro = dendrogram(linkage(x_train, method='ward'))


# In[40]:


# Print out dendrogram which visualize the data stucture and cluster numbers
dendro = dendrogram(linkage(x_train, method='complete', metric="cosine"))


# In[ ]:




