#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import contractions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train = train.sample(frac = 0.5, random_state=42)
train.info()


# In[4]:


# Plotting songs count by genre for both datasets
plt.figure(figsize=(7,7))
plt.subplot(211)
sns.countplot(x='Class Index', data=train, color='skyblue')
plt.title('News Count by Index in Training Data')
plt.ylabel('Number of News', fontsize=12)
plt.xlabel('Index', fontsize=12)
plt.show()


# In[5]:


# Rename the 'Class Index' columns
train = train.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)
test = test.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)


# In[6]:


train['Class_Index'].value_counts()


# In[7]:


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


# In[8]:


train["Description"] = train["Description"].apply(clean)
test["Description"] = test["Description"].apply(clean)


# In[9]:


vectorizer = TfidfVectorizer(sublinear_tf= True, ngram_range=(1,1),
                             stop_words='english',max_features=1200)


# In[10]:


Train_labels = train.Class_Index
Test_labels = test.Class_Index


# In[11]:


Train_text = vectorizer.fit_transform(train['Description'].astype(str))
vectorized_train_text = vectorizer.fit_transform(train['Description']).todense()

Test_text = vectorizer.transform(test['Description'].astype(str))
vectorized_test_text = vectorizer.fit_transform(test['Description']).todense()


# ## Support Vector Machine

# In[12]:


# Creat model object
svm= make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    SVC()
)
# Get the parameters of the model
svm.get_params().keys()


# In[13]:


# Define the parameter grid
svm_params = {
    'pca__n_components': [30, 50],
    'svc__C': [0.1, 1, 0.2],
    'svc__gamma':[0.1, 0.01],
#     'svc__max_iter':[10, 20]
    }


# In[14]:


# Initiating grid search object
svm_model = GridSearchCV(svm, svm_params, cv=3, n_jobs=1, verbose=3)
svm_model.fit(vectorized_train_text, Train_labels)


# In[15]:


#Printing out the best score
svm_model.best_score_


# In[16]:


#Printing out the best parameters
svm_model.best_params_


# In[17]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(svm_model, vectorized_test_text, Test_labels, cmap=plt.cm.Blues, ax=ax)
ax.set_title('Support vector Machine Confusion Matrix for Test Data')


# In[18]:


# Model scor on test data
svc_score = svm_model.score(vectorized_test_text, Test_labels)
svc_score


# In[ ]:




