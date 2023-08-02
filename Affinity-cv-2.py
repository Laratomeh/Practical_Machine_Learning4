#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
import nltk
import re
import contractions
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, make_scorer


# In[2]:


# Reading files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


# Using part of the data due to the very large size,
# and the inability to use it completely due to the small size of the memory. 
train = train.sample(frac = 0.01,  random_state=42)
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


# Define the Count Vectorizer object
vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', min_df=3, max_features=1500)

# Train and test vectors 
train_vectors = vectorizer.fit_transform(train.tokenized)
test_vectors = vectorizer.transform(test.tokenized)

# Train and test labels
Train_labels = train.Class_Index
Test_labels = test.Class_Index


# In[9]:


# Run the model with different random state values
for x, y in zip(range(1, 6, 1), range(10, 61, 10)):
    affinity = AffinityPropagation(random_state=y)
    affinity.fit(train_vectors)
    preds = affinity.fit_predict(test_vectors)  
    # Compute metrics
    adjusted_rand = adjusted_rand_score(Test_labels, preds)
    adjusted_mutual = adjusted_mutual_info_score(Test_labels, preds)
    normalized_mutual = normalized_mutual_info_score(Test_labels, preds)
    # Print the values
    print(x,':', 'Random_State:',y,'Adjusted_Rand:', '%.6f'% adjusted_rand,
          'Adjusted_Mutual:','%.6f'% adjusted_mutual,'Normalized_Mutual:','%.6f'% normalized_mutual)


# In[ ]:




