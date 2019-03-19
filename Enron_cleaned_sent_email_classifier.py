
# coding: utf-8

# In[ ]:


#Need to call important libraries and modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[ ]:


#set the directory first from where we can get the data
os.chdir('D:/vmware')


# In[ ]:


#Lets read the dataset in python using pandas
corpus = pd.read_csv('enron_cleaned_sent_emails.csv')


# In[ ]:


#Check if successfully imported
corpus.head(10)


# In[ ]:


corpus.tail(5)


# In[ ]:


#splitting the file column into 3 more attributes
df= corpus['file'].str.split("/", n=2, expand = True)
corpus['sender_name'] = df[0]
corpus['mail_type'] = df[1]
corpus['file_number'] = df[2]


# In[ ]:


#checking the form
corpus.head(5)


# In[ ]:


corpus = corpus.loc[:,['body','sender_name']]
corpus.head(5)


# In[ ]:


#Top 10 sender list
temp = corpus['sender_name'].value_counts()
temp.head(10).plot(kind='bar')


# In[ ]:


#DTAT PREPROCESSING
#converting everything into NLP format using NLTK package
#removing blanck rows if any
corpus['body'].dropna(inplace = True)


# In[ ]:


#change to lowercase
corpus['body'] = [i.lower() for i in corpus['body']]


# In[ ]:


#Tokenizing to words
corpus['body'] = [word_tokenize(i) for i in corpus['body']]


# In[ ]:


#removing stopwords/Stemmimng/Lemmatization
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,i in enumerate(corpus['body']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(i):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
            
corpus.loc[index,'text_final'] = str(Final_words)


# In[ ]:


#Prepare Train and Test Dataset
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    corpus['text_final'],corpus['sender_name'],test_size=0.3,random_state =0)


# In[ ]:


#Encoding the target variable
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[ ]:


print(Tfidf_vect.vocabulary_)


# In[ ]:


print(Train_X_Tfidf)


# In[ ]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# In[ ]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

