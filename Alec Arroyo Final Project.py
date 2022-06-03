#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Alec Arroyo
#Music Genre Classification

import pandas as pd
import nltk
from pandasql import sqldf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[4]:


fileartist = pd.read_csv('/Users/alec_arroyo/Documents/Sryacuse Data Science Courses/Text Mining/artists-data.csv')


# In[5]:


fileartist[:20]


# In[6]:


filelyrics = pd.read_csv('/Users/alec_arroyo/Documents/Sryacuse Data Science Courses/Text Mining/lyrics-data.csv')


# In[7]:


filelyrics[:20]


# In[8]:


#Merge into 1 DataFrame

len(filelyrics)


# In[9]:


len(fileartist)


# In[10]:


#query = """
#SELECT Genre, COUNT(*) as Total_Count
#FROM fileartist
#GROUP BY Genre
#ORDER BY Total_Count asc
#"""

query1 = sqldf('SELECT * FROM fileartist WHERE Genre = "Rock"')
query2 = sqldf('SELECT * FROM fileartist WHERE Genre = "Funk Carioca"')
query3 = sqldf('SELECT * FROM fileartist WHERE Genre = "Hip Hop"')
query4 = sqldf('SELECT * FROM fileartist WHERE Genre = "Sertanejo"')
query5 = sqldf('SELECT * FROM fileartist WHERE Genre = "Pop"')
query6 = sqldf('SELECT * FROM fileartist WHERE Genre = "Samba"')

totalcounts = sqldf('SELECT Genre, COUNT(*) as Total_Count FROM fileartist GROUP BY Genre ORDER BY Total_Count asc')

#new = sqldf.run(query)


# In[11]:


#How many artists for each genre of music
totalcounts


# In[12]:


#How many artists for each genre of music
data = {'Rock':len(query1), 'Funk Carioca':len(query2), 'Hip Hop':len(query3), 'Sertanejo':len(query4), 'Pop':len(query5), 'Samba':len(query6)}
names = list(data.keys())
values = list(data.values())
plt.title("Num of Artists Per Genre")
plt.bar(names, values)


# In[13]:


len(fileartist['Genre'])


# In[14]:


len(filelyrics)


# In[15]:


#Combine both datasets
finaldata = sqldf('SELECT a.Artist, a.Songs, a.Popularity, b.Sname, b.Lyric, a.Genre, b.Idiom FROM fileartist a JOIN filelyrics b on a.Link = b.ALink GROUP BY b.SLink ORDER BY SName')

finaldata


# In[16]:


#Get rid of None field
testdata = finaldata.dropna()


# In[17]:


len(testdata)


# In[ ]:





# In[18]:


#How many songs for each genre of music UNBALANCED
songct = sqldf('SELECT Genre, COUNT(*) as Total_Count FROM testdata GROUP BY Genre')

songct


# In[19]:


#create table for How many songs for each genre of music UNBALANCED
names = list(songct['Genre'])
values = list(songct['Total_Count'])

plt.title("Num of Songs Per Genre")
plt.bar(names, values)


# In[20]:


#Get sum of popularity by genre of music
query10 = sqldf('select Songs, Genre, Popularity from testdata order by Popularity desc')


# In[21]:


#Get sum of popularity by genre of music
querypop = sqldf('select sum(Popularity) as sum_pop, Genre from query10 group by genre')


# In[22]:


display(querypop)


# In[23]:


names = list(querypop['Genre'])
values = list(querypop['sum_pop'])

plt.title("Popularity Score Per Genre of Music")
plt.bar(names, values)


# In[ ]:





# In[ ]:





# In[24]:


#Time to preprocess data. Function created to do so


# In[25]:


def vectorize(keywords):

    #------------------------------------------------------------------------------------------------------------
    ## Lowercase Words
    #------------------------------------------------------------------------------------------------------------
    
    keywords = [w.lower( ) for w in keywords]
    #print(keywords[:10])
    
    #get count of total words
    print("total words before vectorization: ", len(keywords))
    
    #---------------------------
    ## Add Stemming
    #------------------------------
    
    #Time to stem words together    
    ps = PorterStemmer()   ## method from nltk
    
    stemmed_words=[]  ## make new empty list
    for w in keywords:
        stemmed_words.append(ps.stem(w))
        
    #get count of total words after stemming
    print("total words after stemming (Should be the same): ", len(stemmed_words))
        
        
    #------------------------------------------------------------------------------------------------------------
    ## Removing Stopwords
    #------------------------------------------------------------------------------------------------------------
    
    #Get NLTK stop words
    stop_words=set(stopwords.words("english"))
    
    stop_words_span = set(stopwords.words("spanish"))
    
    stop_words_port = set(stopwords.words("portuguese"))
    
    stop_words_germ = set(stopwords.words("german"))
    
    morestopwords = set(['\'s', '\'nt', 'nt\'', 'nt', 'i', 'n\'t', '\'m', '\'ll', '\'re', 'ca', '\'ve', 'oh', 'yeah', 'a', 'e', 'i', 'o', 'u', 'im', 'thi', 'hi', '\'na', 'na\'', 'na', 'n\'a'])
    
    stop_words = stop_words.union(morestopwords)
    
    stop_words = stop_words.union(stop_words_span)
    
    stop_words = stop_words.union(stop_words_port)
    
    stop_words = stop_words.union(stop_words_germ)
     
    #print(stop_words)
    
    filtered_text=[]   ## Create a new empty list
    
    for w in stemmed_words:
        #print(w)
        if w not in stop_words:
            filtered_text.append(w)
        
    #get count of total words after removing stopwords
    print("total words after removing stopwords: ", len(filtered_text))
    
    
    #------------------------------------------------------------------------------------------------------------
    ## Remove Punctuation
    #------------------------------------------------------------------------------------------------------------
    
    #Check punctuation
    #print(string.punctuation)
    
    #Any reference to punctuation turn into blank
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in filtered_text]
    #print(stripped[:100])
    
    #get count of total words after removing punctuation
    #print("total words after removing punctuation: ", len(stripped))
    
    #------------------------------------------------------------------------------------------------------------
    ## Remove Empty Strings
    #------------------------------------------------------------------------------------------------------------
    
    #remove empty strings 
    while("" in stripped) :
        stripped.remove("")
        
    #get count of total words after removing empty strings
    print("total words after removing punctuation/empty strings: ", len(stripped))
        
    return(stripped)
    


# In[26]:


testdata['Lyric'] = vectorize(testdata['Lyric'])


# In[27]:


display(testdata)


# In[ ]:





# In[28]:


X = testdata['Lyric'].values
Y = testdata['Genre'].values


# In[29]:


#use bayes to train adn test using CV

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
print(X_train[0])
print(Y_train[0])
print(X_test[0])
print(Y_test[0])


# In[30]:


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[31]:


# Check how many training examples in each category
unique, counts = np.unique(Y_train, return_counts=True)
print(np.asarray((unique, counts)))


# In[32]:


#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)

# use the vocabulary constructed from the training data to vectorize the test data. 
X_test_vec = unigram_bool_vectorizer.transform(X_test)


# In[33]:


# import the MNB module
from sklearn.naive_bayes import MultinomialNB

# initialize the MNB model
nb_clf= MultinomialNB()

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,Y_train)


# In[34]:


# test the classifier on the test data set, print accuracy score
nb_clf.score(X_test_vec,Y_test)


# In[35]:


#predict NB models
Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vec)


# In[36]:


# cross validation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, Y_test, Y_pred, cv=3)
avg=sum(scores)/len(scores)
print(avg)


# In[37]:


# print accuracies and classification report
from sklearn.metrics import classification_report
target_names = ["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]
print(classification_report(Y_test, Y_pred, target_names=target_names))


# In[38]:


#confusion martix
df = pd.DataFrame(
    confusion_matrix(Y_test, Y_pred, labels=["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]),
    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], 
    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']
)
print(df)


# In[ ]:





# In[ ]:





# In[37]:


# import the LinearSVC module
from sklearn.svm import LinearSVC

# initialize the LinearSVC model
svm_clf = LinearSVC(C=1)

# use the training data to train the model
svm_clf.fit(X_train_vec,Y_train)


# In[38]:


# test the classifier on the test data set, print accuracy score
svm_clf.score(X_test_vec,Y_test)


# In[39]:


#predict the SVM model
Y_pred = svm_clf.fit(X_train_vec, Y_train).predict(X_test_vec)


# In[40]:


# print confusion matrix and classification report
from sklearn.metrics import classification_report
target_names = ["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]
print(classification_report(Y_test, Y_pred, target_names=target_names))


# In[41]:


# cross validation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
svm_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('svm', LinearSVC(C=1))])
scores = cross_val_score(svm_clf_pipe, Y_test, Y_pred, cv=3)
avg=sum(scores)/len(scores)
print(avg)


# In[42]:


#confusion martix
df = pd.DataFrame(
    confusion_matrix(Y_test, Y_pred, labels=["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]),
    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], 
    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']
)
print(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


#perform Oversampling to deal w/ unbalanced data


# In[44]:


newbalanceddata = testdata


# In[45]:


bal_X = pd.DataFrame(newbalanceddata.Lyric)
bal_Y = pd.DataFrame(newbalanceddata.Genre)


# In[46]:


sqldf('select count(*), Genre from bal_Y group by Genre')


# In[47]:


#most important part
ran = RandomOverSampler(sampling_strategy = 'not majority')


# In[48]:


ran_X, ran_Y = ran.fit_resample(bal_X, bal_Y)


# In[49]:


sqldf('select count(*), Genre from ran_Y group by Genre')


# In[2]:


#use bayes to train and test BALANCED DATASET

ran_X = ran_X['Lyric'].values
ran_Y = ran_Y['Genre'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ran_X, ran_Y, test_size=0.4, random_state=0)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# Check how many training examples in each category
unique, counts = np.unique(Y_train, return_counts=True)
print(np.asarray((unique, counts)))


#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)

# use the vocabulary constructed from the training data to vectorize the test data. 
X_test_vec = unigram_bool_vectorizer.transform(X_test)


# import the MNB module
from sklearn.naive_bayes import MultinomialNB

# initialize the MNB model
nb_clf= MultinomialNB()

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,Y_train)


# test the classifier on the test data set, print accuracy score
print(nb_clf.score(X_test_vec,Y_test))

#predict the NB model
Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vec)


# print confusion matrix and classification report

from sklearn.metrics import classification_report
target_names = ["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]
print(classification_report(Y_test, Y_pred, target_names=target_names))


#confusion martix
df = pd.DataFrame(
    confusion_matrix(Y_test, Y_pred, labels=["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]),
    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], 
    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']
)
print(df)


# In[ ]:





# In[51]:


#repeat same process for SVM

newbalanceddata = testdata

bal_X = pd.DataFrame(newbalanceddata.Lyric)
bal_Y = pd.DataFrame(newbalanceddata.Genre)

#most important part
ran = RandomOverSampler(sampling_strategy = 'not majority')

ran_X, ran_Y = ran.fit_resample(bal_X, bal_Y)


# In[52]:


#use SVM to train and test BALANCED DATASET

ran_X = ran_X['Lyric'].values
ran_Y = ran_Y['Genre'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ran_X, ran_Y, test_size=0.4, random_state=0)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)



# Check how many training examples in each category
unique, counts = np.unique(Y_train, return_counts=True)
print(np.asarray((unique, counts)))


#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)

# use the vocabulary constructed from the training data to vectorize the test data. 
X_test_vec = unigram_bool_vectorizer.transform(X_test)



# import the LinearSVC module
from sklearn.svm import LinearSVC

# initialize the LinearSVC model
svm_clf = LinearSVC(C=1)

# use the training data to train the model
svm_clf.fit(X_train_vec,Y_train)


# test the classifier on the test data set, print accuracy score
print(svm_clf.score(X_test_vec,Y_test))


#predict the SVM model
Y_pred = svm_clf.fit(X_train_vec, Y_train).predict(X_test_vec)


# print confusion matrix and classification report
from sklearn.metrics import classification_report
target_names = ["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]
print(classification_report(Y_test, Y_pred, target_names=target_names))


#confusion martix
df = pd.DataFrame(
    confusion_matrix(Y_test, Y_pred, labels=["Samba", "Funk Carioca", "Hip Hop", "Sertanejo", "Pop", "Rock"]),
    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], 
    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']
)
print(df)


# In[17]:





# In[19]:


newArray = ['Buddy, youre a boy, make a big noise Playing in the street, gonna be a big man someday You got mud on your face, you big disgrace Kicking your can all over the place, singin We will, we will rock you We will, we will rock you Buddy, youre a young man, hard man Shouting in the street, gonna take on the world someday You got blood on your face, you big disgrace Waving your banner all over the place We will, we will rock you, sing it! We will, we will rock you, yeah Buddy, youre an old man, poor man Pleading with your eyes, gonna get you some peace someday You got mud on your face, big disgrace Somebody better put you back into your place, do it! We will, we will rock you, yeah, yeah, come on We will, we will rock you, alright, louder! We will, we will rock you, one more time We will, we will rock you Yeah']


# In[ ]:





# In[ ]:





# In[ ]:


#Intake new song and predict what genre of music it is


# In[20]:


newArray


# In[29]:


#Transform to fit model
X_test_vecARRAY = unigram_bool_vectorizer.transform(newArray)


# In[31]:


#predict Lyrics to genre
Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vecARRAY)


# In[32]:


#Genre prediction was correct. this is a rock song
Y_pred


# In[ ]:





# In[ ]:





# In[33]:


#Prompt user to input song lyrics
newlyric = input("Enter your song lyrics for one song here: ")


# In[40]:


#Convert lyrics variable into a list
newlyric1 = []
newlyric1 = [newlyric]


# In[42]:


#Transform to fit model
X_test_vecARRAY = unigram_bool_vectorizer.transform(newlyric1)


# In[43]:


#predict Lyrics to genre
Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vecARRAY)


# In[44]:


#Genre prediction was correct. this is a Hip Hop song
Y_pred


# In[ ]:




