#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import re
import numpy as np
import pandas as pd
import contractions
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, make_scorer, adjusted_mutual_info_score 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


# In[ ]:


# Reading files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[ ]:


# Using part of the data due to the very large size,
# and the inability to use it completely due to the small size of the memory. 
train = train.sample(frac = 0.2,  random_state=42)
train.info()


# In[ ]:


# Rename the 'Class Index' columns
train = train.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)
test = test.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)


# In[ ]:


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
#     lemmatizer = WordNetLemmatizer()
#     text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


# In[ ]:


# Apply function to clean the text
train["Description"] = train["Description"].apply(clean)
test["Description"] = test["Description"].apply(clean)


# In[ ]:


# Define the TF-IDF vectorizer object
vectorizer = TfidfVectorizer(stop_words='english', norm='l2', max_features=1000, sublinear_tf= True, ngram_range=(1,1))

# Train and test vectors 
Train_text = vectorizer.fit_transform(train['Description'].astype(str))
vectorized_train_text = vectorizer.fit_transform(train['Description']).todense()
Test_text = vectorizer.transform(test['Description'].astype(str))
vectorized_test_text = vectorizer.fit_transform(test['Description']).todense()

# Train and test labels
Train_labels = train.Class_Index
Test_labels = test.Class_Index


# ## AffinityPropagation

# In[ ]:


# Creat model object
mp = make_pipeline(
    PCA(random_state=42),
    AffinityPropagation()
)
# Get the parameters of the model
mp.get_params().keys()


# In[ ]:


# Define the parameter grid
param_grid = {
    'pca__n_components': [100,200],
    'affinitypropagation__damping': [0.8, 0.9],
    'affinitypropagation__preference': [-10, -40],
    'affinitypropagation__max_iter':[100, 500]
    }


# In[ ]:


# Create an instance of GridSearchCV
clf = GridSearchCV(mp, param_grid = param_grid, cv=3, scoring = adjusted_mutual_info_score, verbose = 2)

# Fit the GridSearchCV instance to the text data
clustering = clf.fit(vectorized_train_text)


# In[ ]:


# Predict cluster labels for test data
pred_labels = clustering.predict(vectorized_test_text)

# Compute silhouette score for test data
test_silhouette_score = silhouette_score(vectorized_test_text, Test_labels, metric='euclidean')
print("Silhouette score for test data: ", test_silhouette_score)


# In[ ]:


# Compute the metrics
rand_score = rand_score(Test_labels, pred_labels)
adjusted_rand = adjusted_rand_score(Test_labels, pred_labels)
adjusted_mutual = adjusted_mutual_info_score(Test_labels, pred_labels)
normalized_mutual = normalized_mutual_info_score(Test_labels, pred_labels)
silhouette_score = silhouette_score(vectorized_test_text, Test_labels, metric='euclidean')

