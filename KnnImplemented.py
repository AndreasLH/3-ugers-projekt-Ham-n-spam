import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def prepare_data(csv_file):
    """ Function: Loads data and prepares data (feature transformation)
        Input: cvs-file with messages and class (spam = 1 or not spam = 0 )
        Output: pd-DataFrame-object
    """
    
    # Get raw data and remove unnessecary information from text
    data = pd.read_csv(csv_file)
    data['text'] = data['text'].str.strip('Subject:')
    data['text'] = data['text'].str.strip('fwd :')
    data['text'] = data['text'].str.strip(' re ')
    
    # Load stop-words, define tokenizer and stemmer 
    stop_words = set(stopwords.words('english'))     # https://onlinecoursetutorials.com/nlp/how-to-remove-punctuation-in-python-nltk/
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()

    numb_mails = len(data.text)  
    
    # Remove counter after debugging 
    count = 0

    # Loops through each message
    for message in range(numb_mails): # Remove '-5000' after debugging
        # make message lower case
        text = data.text[message].lower()
        
        # Substitute special tokens with describtive strings 
        text = text.replace('$', 'DOLLAR')
        text = text.replace('@', 'EMAILADRESS') # problem ift. " @ "
        
        # Tokenize + remove punctuation (Problem: @-sign is also removed) 
        tokens1 = tokenizer.tokenize(text)        
        
        # Remove stop-words 
        tokens2 = [w for w in tokens1 if not w in stop_words]
        
        # Stemming tokens   
        numb_tokens = len(tokens2)
        
        for token in range(numb_tokens):
            tokens2[token] = ps.stem(tokens2[token])
             
        # Sustitute number (special token) with 'NUMBER' (numbers can be split by with space)
        for token in range(numb_tokens):
             try:
                 int(tokens2[token]) 
                 tokens2[token] = "NUMBER"
             except:
                 pass      
        
        last_token = ""
        for token in reversed(range(numb_tokens)):
            if (last_token == tokens2[token]) and (last_token=='NUMBER'):
                del tokens2[token+1]
            
            last_token = tokens2[token]

        # Collect tokens to string and assign to dataframe 
        prepared_string = " ".join(tokens2)
        
        data.at[message,'text'] = prepared_string 
        # Remove after debugging 
# =============================================================================
#         if count%50==0:
#             print(count)
#         count += 1
# =============================================================================
    
    return data 
        
result = prepare_data('emails.csv')
print(result.text)

x = result.text
y = result.spam

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# =============================================================================
# from sklearn.feature_extraction.text import CountVectorizer
# #dette er bag of words!
# vect = CountVectorizer()
# vect.fit(X_train)
# X_train_dtm = vect.transform(X_train)
# #tilsvarende til de 2 forrige linjer (fit og transform)
# #X_train_dtm = vect.fit_transform(X_train)
# print(X_train_dtm.shape)
# X_test_dtm = vect.transform(X_test)
# print(X_test_dtm.shape)
# =============================================================================

#TF-IDF implementation
from sklearn.feature_extraction.text import TfidfVectorizer
#dette er bag of words!
vect = TfidfVectorizer()
X_train_dtm = vect.fit_transform(X_train)

print(X_train_dtm.shape)
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape)

#make model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
#accuracy
acc = metrics.accuracy_score(y_test, y_pred_class)
print(acc)
#confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred_class)
print(conf_mat)
#confusion matrix
# format [true neg, false pos]
#        [false neg, true pos]
# positive class is spam


#false positive messages
print(X_test[y_pred_class > y_test])
#false negative messages
print(X_test[y_pred_class < y_test])

#naive bayes er ikke så præcis når den siger at det med 100% er spam, det er bare udregnet
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)
#area under curve
AUC = metrics.roc_auc_score(y_test, y_pred_prob)
print(AUC)

#Make KNN model
from sklearn.neighbors import KNeighborsClassifier
#The number of neighbors are defined aka the K value - 5 is used since it is the most common K value for the algorihm
KNN = KNeighborsClassifier(n_neighbors = 5)
#The data is trained
KNN.fit(X_train_dtm,y_train)
#Predictions is used
y_pred_KNN = KNN.predict(X_test_dtm)


from sklearn.metrics import classification_report,confusion_matrix
#The confusion matrix is calculated 
KNNconMat = confusion_matrix(y_test,y_pred_KNN)
#Class representation for the KNN algorithm
KNNClassRep = classification_report(y_test,y_pred_KNN)

print('KNN confusion matrix ', KNNconMat)
print('KNN classification reprensentation ', KNNClassRep) 

import matplotlib.pyplot as plt
error = []

for i in range(1,10):
    KnN = KNeighborsClassifier(n_neighbors=i)
    KnN.fit(X_train_dtm,y_train)
    pred_i_knn = KnN.predict(X_test_dtm)
    error.append(np.mean(pred_i_knn != y_test))
    
plt.figure(figsize = (15,15))
plt.plot(range(1,10),error,color = '#cc338b',linestyle = 'solid',
         marker = 'o',markerfacecolor = '#8bbe1b',markersize = 20)
plt.title('Error Rate K value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')



