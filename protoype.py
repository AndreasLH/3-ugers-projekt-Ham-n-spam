# -*- coding: utf-8 -*- | Wed Jan  8 12:15:29 2020 | Andreas Lau Hansen
#based on Tutorial: Machine Learning with Text in scikit-learn

import pandas as pd
import numpy as np

df = pd.read_csv('emails.csv')

x = df.text
y = df.spam

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


"""
arr = np.array(df)

simple_train = arr[0, 0]
simple_train = list(arr[0, 0])
simple_train = list(arr[:, 0])

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

vect.fit(simple_train)
a = vect.get_feature_names()

simple_train_dtm = vect.transform(simple_train)
simple_train_dtm.toarray()
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
print(simple_train_dtm)

simple_test = ["hello please respond to my email"]
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
"""