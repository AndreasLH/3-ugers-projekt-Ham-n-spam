# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:09:37 2020

@author: Kirstine Cort Graae
"""

"""https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/"""

#Import libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

#Define the function
def k_error_range(X_train_dtm, y_train, X_test_dtm, y_test):
    
    #make an empty array for the errors
    error = []
    #Loop through the k values from 1 to 25
    for k in range(1,25):
        
        #The K- nearest neighbors are calculated
        KnN = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', n_jobs = -1)
        
        #The data is used for training
        KnN.fit(X_train_dtm,y_train)
        
        #Make the predictions using the KNN model
        pred_k_knn = KnN.predict(X_test_dtm)
        
        #Append the mean error of the predection is calculated to the error list
        error.append(np.mean(pred_k_knn != y_test))
        
    #Plot  
    plt.figure(dpi = 600)
    plt.plot(range(1,25),error,color = '#571B7E',linestyle = 'solid',linewidth = 3, 
             marker = 'D',markerfacecolor = '#C25283')
    plt.title('Fejl for k[1;25]',fontsize = 30)
    plt.xlabel('K-værdi',fontsize = 20)
    plt.ylabel('Standard Fejl',fontsize = 20)
    plt.savefig('error_fig_TFIDF_5050')
    plt.show()
    
def k_accuracy_range(X_train_dtm, y_train, X_test_dtm, y_test):
    
    #Make empty list of scores
    accuracy_list = []
    
    #Loop through the first the K values from 1 to 25
    for k in range(1,25):
        
        #The K-  Nearest Neighbors are calculated
        knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', n_jobs = -1)
        
        #The data is trained
        knn.fit(X_train_dtm,y_train)
        
        #The predictions are made
        y_pred_knn= knn.predict(X_test_dtm)
        
        #The accuracy of the prediction is calculated and appended to the accuracy list
        accuracy_list.append(metrics.accuracy_score(y_test,y_pred_knn))
        
    #Plot
    plt.figure(dpi = 600)
    plt.plot(range(1,25),accuracy_list,color = '#571B7E',linestyle = 'solid',linewidth = 3,
             marker = 'D',markerfacecolor = '#C25283')
    plt.title('Nøjagtighed for k[1;25]',fontsize = 30)
    plt.xlabel('K-værdi',fontsize = 20)
    plt.ylabel('Test præcision',fontsize = 20)
    plt.savefig('accuracy_fig_TFIDF_5050')
    plt.show()
    


