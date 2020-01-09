# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:28:24 2020

@author: Louis Thygesen
"""

# Import typical libraries 
import pandas as pd

# Import SciKit modules 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Get data and store as Pandas series (1D-array)
df = pd.read_csv('processed_emails.csv')

x = df.text
y = df.spam

# Split dataset into training- and testset (.50/.50)
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=1)  # still pandas series 

# Create matrix of type DxW (where D is documents in training data and W is word in vocabulary)
# Attention 1: All feature transformation in TfidfVectorizer has {default=none}
#              so our own feature transformation isn't changed!
# Attention 2: Returns sparse matrix - Ex: X_train_dtm[0] returns list of document 
#              number (0), word number (ex: 1857) and word tf-idf-score 
vect = TfidfVectorizer()

# The feature-set is chossen as the vocabulary in training-data (see fit-part)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Convert sparse matrix to array (matrix)
X_train_dtm = X_train_dtm.toarray()
X_test_dtm = X_test_dtm.toarray()

# Create ML-model (NB Gauss) and fit to training-set (calculate pobabilities)
gnb = GaussianNB()
gnb.fit(X_train_dtm, y_train)

# Classification of test-set (emails)
y_pred_class = gnb.predict(X_test_dtm)
y_true_class = y_test 

# Create confussion matrix 
conf_matrix = confusion_matrix(y_true_class, y_pred_class)
print(conf_matrix)

# Compute model-accuracy
accuracy = accuracy_score(y_true_class, y_pred_class)
print(accuracy)

# Print spam messages falsly identified as regular emails (and id-number)
print(X_test[y_pred_class > y_true_class])

# Print regular emails falsly identified as spam (and id-number)
print(X_test[y_pred_class < y_true_class])


#naive bayes er ikke så præcis når den siger at det med 100% er spam, det er bare udregnet
y_pred_prob = gnb.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)
#area under curve
AUC = roc_auc_score(y_true_class, y_pred_prob)
print(AUC)


# Debugging
# =============================================================================
# # Prints amounts of emails in each dataset 
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# =============================================================================

# -------------------- Ikke benytede dele af kode -----------------------------
# Part 1: Data transformation 
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
# 
# Part 2: Test Tf-idf-vectorizer
# =============================================================================
# print(type(X_train_dtm[0]))
#
# print(X_train_dtm.shape)   #Indeholder +27.000 ord 
# print(X_test_dtm.shape)    #Indeholder det præcis det samme 
# =============================================================================
