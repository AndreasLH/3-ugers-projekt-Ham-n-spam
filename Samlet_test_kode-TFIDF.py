# -*- coding: utf-8 -*-
""" Program Describtion:
    A program created to be able to test and export results (data) of
    different ML-algortims under varying circumstances (changing parameters)
"""
"""    This part is for changing variables while testing ML-algortihms     """

###############################################################################
###############################################################################
#                      -----  Mission Control  -----                          #
###############################################################################
filename = 'Datasæt/processed_emails_v1.csv'

# Change parameters (split_dataset, tfidfVectorizer and KNN)
SD_random_state = 1               # None or int pseudo randomness
SD_shuffle = True
SD_train_size = 4000              # None, int or float
SD_test_size = 1728           # None, int or float

TFIDF_max_features = None           # None or int
#number of k in knn
KNN_k = 2

# Change which model to run (GaussNB = 0, MultinomiaNB = 1 KNN = 2)
model = 2

# Train-split mode
split = 1     # (0 = 24/76 and 1 = 50/50)

###############################################################################
###############################################################################


""" This part contains the program (data transformation and ML-algorithms) """

# Import general libraires
import pandas as pd
import numpy as np

# Import SciKit modules (for GaussNB)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
# Import SciKit module (for KKN)
from sklearn.neighbors import KNeighborsClassifier
# Import modules for data representation
from sklearn.metrics import confusion_matrix
# Import train and test dataset
from split5050_alternativ import split5050

def tfidf(X_train, X_test, TFIDF_max_features):
    """ Create TF-IDF-matrix of training- and test-features """
    vect = TfidfVectorizer(max_features = TFIDF_max_features,
                           lowercase = False, analyzer = 'word')

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Convert sparse matrix to array (matrix)
    X_train_dtm = X_train_dtm.toarray()
    X_test_dtm = X_test_dtm.toarray()

    print('shape of features', np.shape(X_test_dtm))

    # Store vocabulary (especially important if max_features != None)
    vocabulary = vect.get_feature_names()

    return X_train_dtm, X_test_dtm, vocabulary


def gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with gaussian Naive Bayes """
# Create ML-model (NB Gauss) and fit to training-set (calculate pobabilities)
    gnb = GaussianNB()
    gnb.fit(X_train_dtm, y_train)
    # Classification of test-set (emails)
    y_pred_class = gnb.predict(X_test_dtm)
    y_true_class = y_test
    return y_pred_class, y_true_class, train_time, test_time

def multinomialNB(X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with gaussian Naive Bayes """
# Create ML-model (NB Multinomial) and fit to training-set
#(calculate pobabilities)
    mnb = MultinomialNB()
    mnb.fit(X_train_dtm, y_train)
    # Classification of test-set (emails)
    y_pred_class = mnb.predict(X_test_dtm)
    y_true_class = y_test
    return y_pred_class, y_true_class, train_time, test_time

def KKN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with K nearest neighbor """
    # Create KNN-model (general)
    KNN = KNeighborsClassifier(n_neighbors = KNN_k,
                               algorithm = 'brute', n_jobs = -1)
    # Fit model to our training data
    KNN.fit(X_train_dtm,y_train)
    # Predict classes for test-documents (using euclidian distance)
    y_pred_class = KNN.predict(X_test_dtm)
    y_true_class = y_test

    return y_pred_class, y_true_class, test_time


# Get data (features and targets) and store as Pandas series (1D-array)
df = pd.read_csv(filename)

x = df.text
y = df.spam

debbuging_accuracy = np.array([]) # Remove after debbugning


# Split data into training- and testset (SciKit) (24/76)
if split == 0:
    kwargs = {
        'train_size': SD_train_size,
        'test_size': SD_test_size,
        'random_state': SD_random_state,
        'shuffle': SD_shuffle,
        'stratify': y
        }

    X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs)

# Split data into training- and testset (SciKit) (50/50)
elif split == 1:
    X_train, X_test, y_train, y_test = split5050(x,y)

# Create TFIDF-matrix (custom function with SciKit)
X_train_dtm, X_test_dtm, vocabulary =\
 tfidf(X_train, X_test, TFIDF_max_features)

# Make prediction (with choosen model)
if model == 0:
    y_pred_class, y_true_class, train_time, test_time =\
    gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test)

elif model == 1:
    y_pred_class, y_true_class, train_time, test_time =\
        multinomialNB(X_train_dtm, y_train, X_test_dtm, y_test)

elif model == 2:
    y_pred_class, y_true_class, test_time =\
        KKN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test)

numb_test = np.size(y_pred_class)
numb_correct = np.size(y_pred_class[y_pred_class == y_true_class])

accuracy = numb_correct/numb_test
print("Accuracy:  {:.4f}".format(accuracy))

# Create and show confussion matrix
conf_matrix = confusion_matrix(y_true_class, y_pred_class)
print("Confusion matrix:")
print(conf_matrix)

print('\n')

# Ekstra statistisk (debugging)
debbuging_accuracy = np.append(debbuging_accuracy, accuracy)
print(conf_matrix[0,0]+conf_matrix[0,1])
print(conf_matrix[1,0]+conf_matrix[1,1])

# Print debugging statistiker
mean = np.mean(debbuging_accuracy)

#lav KNN plots
from KNNgraph import k_error_range
from KNNgraph import k_accuracy_range

k_accuracy_range(X_train_dtm, y_train, X_test_dtm, y_test)
k_error_range(X_train_dtm, y_train, X_test_dtm, y_test)
