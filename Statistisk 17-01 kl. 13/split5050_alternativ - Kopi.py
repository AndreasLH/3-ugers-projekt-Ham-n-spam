# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:37:22 2020
@author: Louis Thygesen
"""

import numpy as np 

def split5050(x,y):
    spam_index = np.array([])
    ham_index = np.array([])
    
    count_spam = 0
    count_ham = 0
    
    for i in range(len(x)):
        if y[i] == 1:
            spam_index = np.append(spam_index, i)
            count_spam += 1
            
        else:
            ham_index = np.append(ham_index, i)
            count_ham += 1
            
    # Spam part of dataset
    spam_text = x.drop(ham_index, axis = 0)    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
    spam_class = y.drop(ham_index, axis = 0)
    
    spam_text.index = np.arange(0,count_spam)    #https://www.geeksforgeeks.org/python-pandas-series-rename/
    spam_class.index = np.arange(0, count_spam)
    
    # Ham part of dataset
    ham_text = x.drop(spam_index, axis = 0)
    ham_class = y.drop(spam_index, axis = 0)
    
    ham_text.index = np.arange(0,count_ham)
    ham_class.index = np.arange(0, count_ham)

    # Make training-dataset of 50/50 (train-size = 1500)
    numb_class = 750
    
    # Make pd.series for training 
    train_spam_text = spam_text.loc[:numb_class-1]     # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
    train_spam_class = spam_class.loc[:numb_class-1]
    
    train_ham_text = ham_text.loc[:numb_class-1]
    train_ham_class = ham_class.loc[:numb_class-1]
    
    X_train = train_spam_text.append(train_ham_text)
    Y_train = train_spam_class.append(train_ham_class)
    
    X_train.index = np.arange(0,1500)
    Y_train.index = np.arange(0,1500)
    
    # Make pd.series for testing
    andel = 0.239 * 2268
    grænse = numb_class + andel
    
    test_spam_text = spam_text.loc[numb_class:grænse-1]
    test_spam_class = spam_class.loc[numb_class:grænse-1]
    
    spam_numb = len(test_spam_class)
    ham_numb = 2268 - spam_numb 
    index = numb_class + ham_numb 
    
    test_ham_text = ham_text.loc[numb_class:index-1]
    test_ham_class = ham_class.loc[numb_class:index-1]
    
    X_test = test_spam_text.append(test_ham_text)
    Y_test = test_spam_class.append(test_ham_class)
    
    X_test.index = np.arange(0,2268)
    Y_test.index = np.arange(0,2268)
    
    return X_train, X_test, Y_train, Y_test 


import pandas as pd
df = pd.read_csv('Datasæt/processed_emails_v1.csv')

x = df.text
y = df.spam


X_train, X_test, Y_train, Y_test = split5050(x,y)


 

        