# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:11:11 2020

@author: Kirstine Cort Graae
"""

""" Program Describtion:
    A program created to be able to test and export results (data) of
    different ML-algortims under varying circumstances (changing parameters)
    Based on the following sources:
    1. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier (KNN)
    2. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB (Gauss NB)
    3. https://scikit-learn.org/stable/modules/naive_bayes.html (Generelt om NB)
    4. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html (Multinomial NB)
    5. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (TfidfVectorizer)
"""


"""    This part is for changing variables while testing ML-algortihms     """

###############################################################################
###############################################################################
#                      -----  Mission Control  -----                          #
###############################################################################
filename = 'processed_email5050_stop_removed.csv'

# Change parameters (split_dataset, tfidfVectorizer and KNN)
SD_random_state = None            # None or int
SD_shuffle = True
SD_train_size = 0.9               # None, int or float
SD_test_size = 0.1             # None, int or float

TFIDF_max_features = None           # None or int

KNN_k = 4

# Change which model to run (GaussNB = 0, MultinomiaNB = 1 KNN = 2)
model = 2

# Number of times test should be maken (with these parameters)
iterations = 5

###############################################################################
###############################################################################


""" This part contains the program (data transformation and ML-algorithms) """

# Import general libraires
import pandas as pd
import numpy as np

# Import runtime libraries
from time import time

# Import SciKit modules (for GaussNB)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Import SciKit module (for KKN)
from sklearn.neighbors import KNeighborsClassifier

# Import modules for data representation
from sklearn.metrics import confusion_matrix

def tfidf(X_train, X_test, TFIDF_max_features):
    """ Create TF-IDF-matrix of training- and test-features """
    vect = TfidfVectorizer(max_features = TFIDF_max_features, lowercase = False, analyzer = 'word')

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Convert sparse matrix to array (matrix)
    X_train_dtm = X_train_dtm.toarray()
    X_test_dtm = X_test_dtm.toarray()

    # Store vocabulary (especially important if max_features != None)
    vocabulary = vect.get_feature_names()

    return X_train_dtm, X_test_dtm, vocabulary


def gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with gaussian Naive Bayes """
    # Create ML-model (NB Gauss) and fit to training-set (calculate pobabilities)
    start_time = time()    # https://stackoverflow.com/a/1557584/12428216

    gnb = GaussianNB()
    gnb.fit(X_train_dtm, y_train)

    # Compute runtime (training)
    end_time = time()
    train_time = end_time - start_time

    # Classification of test-set (emails)
    start_time = time()

    y_pred_class = gnb.predict(X_test_dtm)
    y_true_class = y_test

    # Compute runtime (test)
    end_time = time()
    test_time = end_time - start_time

    return y_pred_class, y_true_class, train_time, test_time

def multinomialNB(X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with gaussian Naive Bayes """
    # Create ML-model (NB Multinomial) and fit to training-set (calculate pobabilities)
    start_time = time()    # https://stackoverflow.com/a/1557584/12428216

    mnb = MultinomialNB()
    mnb.fit(X_train_dtm, y_train)

    # Compute runtime (training)
    end_time = time()
    train_time = end_time - start_time

    # Classification of test-set (emails)
    start_time = time()

    y_pred_class = mnb.predict(X_test_dtm)
    y_true_class = y_test

    # Compute runtime (test)
    end_time = time()
    test_time = end_time - start_time

    return y_pred_class, y_true_class, train_time, test_time


def KKN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with K nearest neighbor """
    # Create KNN-model (general)
    KNN = KNeighborsClassifier(n_neighbors = KNN_k, algorithm = 'brute', n_jobs = -1)

    # Fit model to our training data
    KNN.fit(X_train_dtm,y_train)

    # Predict classes for test-documents (using euclidian distance)
    start_time = time()

    y_pred_class = KNN.predict(X_test_dtm)

    end_time = time()
    test_time = end_time - start_time

    y_true_class = y_test

    return y_pred_class, y_true_class, test_time


# Get data (features and targets) and store as Pandas series (1D-array)
df = pd.read_csv(filename)

x = df.text
y = df.spam



debbuging_accuracy = np.array([]) # Remove after debbugning

for iteration in range(iterations):
    # Split data into training- and testset (SciKit)
    kwargs = {
# https://stackoverflow.com/questions/9539921/how-do-i-create-a-python-function-with-optional-arguments
# https://realpython.com/python-kwargs-and-args/
        'train_size': SD_train_size,
        'test_size': SD_test_size,
        'random_state': SD_random_state
        }

    X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs)

    # Create TFIDF-matrix (custom function with SciKit)

    X_train_dtm, X_test_dtm, vocabulary = tfidf(X_train, X_test, TFIDF_max_features)

    # Make prediction (with choosen model)
    if model == 0:
        y_pred_class, y_true_class, train_time, test_time = gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test)

    elif model == 1:
        y_pred_class, y_true_class, train_time, test_time = gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test)

    elif model == 2:
        y_pred_class, y_true_class, test_time = KKN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test)

    # Compute accuracy (and show)
    print("Iteration: {}".format(iteration+1))

    numb_test = np.size(y_pred_class)
    numb_correct = np.size(y_pred_class[y_pred_class == y_true_class])

    accuracy = numb_correct/numb_test
    print("Accuracy:  {:.4f}".format(accuracy))

    # Show runtime
    if model == 1:
        print("Runtime:   Training ({:.1f} s.) Testing ({:.1f} s.)".format(train_time, test_time))

    else:
        print("Runtime:   Testing ({:.1f} s.)".format(test_time))

    # Show features
    if type(TFIDF_max_features) == int:
        if TFIDF_max_features <= 15:

            print("Feature vocabulary:")

            for word in range(np.size(vocabulary)):
                print("\t{}. {}".format(word+1,vocabulary[word]))

    else:
            print("Feature vocabulary: Too many features!")

    # Create and show confussion matrix
    conf_matrix = confusion_matrix(y_true_class, y_pred_class)
    print("Confusion matrix:")
    print(conf_matrix)

    print('\n')

    # Ekstra statistisk (debugging)
    debbuging_accuracy = np.append(debbuging_accuracy, accuracy)

# Print debugging statistiker
mean = np.mean(debbuging_accuracy)

print("-------- Over all ----------")
print("Mean: {:.4f}".format(mean))

# =============================================================================
# a = pd.DataFrame(X_train_dtm,columns = vocabulary)
# print(a['th'])
# =============================================================================
# =============================================================================
# from KNNgraph import k_error_range
# from KNNgraph import k_accuracy_range
# k_error_range(X_train_dtm, y_train, X_test_dtm, y_test)
# k_accuracy_range(X_train_dtm, y_train, X_test_dtm, y_test)
# =============================================================================
