#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import re
import nltk
import string
import numpy as np
import contractions
import pandas as pd
import tensorflow as tf
from termcolor import colored
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[2]:


# Importing train and test datasets
train_data = pd.read_csv(r'train.csv', engine='python', encoding='utf-8')
test_data = pd.read_csv(r'test.csv', engine='python', encoding='utf-8')


# In[3]:


train_data.head()


# In[4]:


# Rename the 'Class Index' columns
train_data = train_data.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)
test_data = test_data.rename(columns = {'Class Index': 'Class_Index'}, inplace = False)


# In[5]:


# Calculate the index values
train_data['Class_Index'].value_counts() , test_data['Class_Index'].value_counts()


# In[6]:


# Define a function for text cleaning
def clean(text):
    # convert text to lowercase
    text = text.lower()  
    # git red of non word carachters
    text = re.sub(r'\W', ' ', text)
    # remove digits
#     text = re.sub(r'\d', ' ', text)
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
    # get rid of any word which is not in English dictionary
    # text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    # # Stemming words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # Lemmatizing words
    # lemmatizer = WordNetLemmatizer()
    # text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


# In[7]:


# Apply the clean function
train_data["Description"] = train_data["Description"].apply(clean)
test_data["Description"] = test_data["Description"].apply(clean)


# In[8]:


train_data.head()


# In[9]:


# Define training, testing data, and maximum length
X_train = train_data["Description"] 
X_test = test_data["Description"] 
length_max = X_train.map(lambda x: len(x.split())).max()
length_max


# In[10]:


# Define training and testing labels
y_train = train_data['Class_Index'].apply(lambda x: x-1).values
y_test = test_data['Class_Index'].apply(lambda x: x-1).values


# In[12]:


# Splitting the training data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[13]:


# Define the vocabularies size, embedding size
vocabularies = 30000 
embeddings = 40
# Define the tokenizer
tokenizer = Tokenizer(num_words=vocabularies)
tokenizer.fit_on_texts(X_train.values)
# Tokenizing and Padding data
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=length_max)
X_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val, maxlen=length_max)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=length_max)


# In[14]:


# Build the model and print out the model summary
lstm_model = Sequential()
lstm_model.add(Embedding(vocabularies, embeddings, input_length=length_max))
lstm_model.add(Bidirectional(LSTM(128, return_sequences=True))) 
lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
lstm_model.add(Bidirectional(LSTM(32, return_sequences=True)))
lstm_model.add(GlobalMaxPooling1D())
lstm_model.add(Dense(512))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(256))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(128))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(64))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(4, activation='softmax'))
lstm_model.summary()


# In[15]:


class myCallback(tf.keras.callbacks.Callback):
    def epoch(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.95):
            self.lstm_model.stop_training = True
            print("\nThe accuracy has reached > 95%!")
callbacks = myCallback()


# In[16]:


# Compile the model
lstm_model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
# Fit the model
history = lstm_model.fit(X_train, y_train, batch_size=256, epochs=15, validation_data=(X_val, y_val), verbose=1,
                         callbacks=[callbacks], validation_steps=5, steps_per_epoch=25)


# In[17]:


print('Training Accuracy')
accuracy=lstm_model.evaluate(X_train,y_train)
print('Loss: {} \nAccuracy: {}'.format(accuracy[0],accuracy[1]))


# In[18]:


print('Validation Accuracy')
accuracy=lstm_model.evaluate(X_val,y_val)
print('Loss: {} \nAccuracy: {}'.format(accuracy[0],accuracy[1]))


# In[19]:


print('Test Accuracy')
accuracy=lstm_model.evaluate(X_test,y_test)
print('Loss: {} \nAccuracy: {}'.format(accuracy[0],accuracy[1]))


# In[20]:


# Plot training and validation loss and accuracy
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, 'b', label='Train accuracy')
    plt.plot(val_acc, 'g', label='Validation accuracy')
    plt.title('Train and validation accuracy')
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, 'b', label='Train loss')
    plt.plot(val_loss, 'g', label='Validation loss')
    plt.title('Train and validation loss')
    plt.xlabel("Epochs")
    plt.legend()
    
plot_history(history)


# In[21]:


# Save model for later usage
lstm_model.save('LSTM_model.hdf5')


# In[22]:


# Define a function to predict new texts
def predict(text):
    labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
    seq = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=length_max)
    preds = [labels[np.argmax(i)] for i in lstm_model.predict(seq)]
    for news, label in zip(text, preds):
        print('{} - {}'.format(colored(news, 'blue'), colored(label, 'green')))


# In[23]:


predict(["These 6 Club stocks look reasonably priced as Wall Street shuns high flyers"])


# In[24]:


predict(["Russian President Vladimir Putin had proposed a 36-hour truce to mark Russian Orthodox Christmas, but Kyiv dismissed it as a ploy."])


# In[29]:


predict(["In the future all machines, from milling machines to welding robots, will be networked with one another, according to German research organization Fraunhofer. "])


# In[27]:


predict(["Saudi Arabian club Al Nassr have denied reports that Cristiano Ronaldo has a clause in his contract where the Portugal forward is to serve as an ambassador for the Gulf country's 2030 World Cup bid."])


# In[30]:


# Plotting confusion matrix for Test set
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
preds = [np.argmax(i) for i in lstm_model.predict(X_test)]
cm  = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(cmap=plt.cm.Blues)
plt.xticks(range(4), labels, fontsize=6)
plt.yticks(range(4), labels, fontsize=6)
plt.show()


# In[31]:


cm  = confusion_matrix(y_test, preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels).plot(cmap=plt.cm.Blues)
plt.xticks(range(4), labels, fontsize=6)
plt.yticks(range(4), labels, fontsize=6)
plt.show()


# In[32]:


print('Classification Report for Test set')
print(classification_report(y_test, preds))


# In[ ]:




