# -*- coding: utf-8 -*-
"""
Program Description:
A program created to be able to test and export results (data) of
different ML-algortims under varying circumstances (changing parameters)
Based on the following sources:
"""
# Import general libraires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import runtime libraries
from time import time
# Import SciKit modules (for GaussNB)
from sklearn.model_selection import train_test_split
#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
#bow
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
# Import SciKit module (for KNN)
from sklearn.neighbors import KNeighborsClassifier
# Import modules for data representation
from sklearn.metrics import confusion_matrix
from barplot import confusion_matrix_visual

"""    This part is for changing variables while testing ML-algortihms     """

###############################################################################
#                      -----  Mission Control  -----                          #
###############################################################################
#csv file in
filename = 'datasæt/processed_emails_v1.csv'
#png (plot) file out name
file_out = 'plot'
#show plots or not
show_plots = True

n_samples = 4
train_size_vect = np.linspace(3, 2500, n_samples, dtype = 'int32')
max_features_vect = np.linspace(100, 20_000, n_samples, dtype = 'int32')
number_neighbors_vect = np.linspace(1, 40, n_samples, dtype = 'int32')
#feature to plot
model_param = train_size_vect

# Change parameters (split_dataset, tfidfVectorizer and KNN)
SD_random_state = 1            # None or int
SD_shuffle = True
SD_train_size = 1500                # None, int or float
SD_test_size = 2268            # None, int or float

TFIDF_max_features = None       # None or int

KNN_k = 2

# Change which model to run (GaussNB = 0, MultinomialNB = 1, KNN = 2)
model = 1
# use TF-IDF = True or BOW = False
tfidf_vec = False

# Number of times test should be maken (with these parameters)
iterations = 1

BOW_max_features = None              # None or int

# Get data (features and targets) and store as Pandas series (1D-array)
df = pd.read_csv(filename)
x = df.text
y = df.spam

debugging_accuracy = np.array([]) # used to calculate mean
output = np.zeros(n_samples)
confint = np.zeros(n_samples)

###############################################################################
#                               Functions                                     #
###############################################################################
def plot(model_param):
    plt.figure()
    plt.fill_between(model_param, output-confint,output+confint,
                     color = 'gray',alpha = 0.2)
    plt.plot(model_param, output, 'k')
    #use latex font for graph
    plt.rc('text', usetex=True)
    plt.title('Accuracy as a function of train size')
    plt.xlabel('Train size')
    plt.ylabel('Accuracy \%')
    #plt.text(1100, 0.72, f'95\% CI with a test size of {SD_test_size}')
    plt.ylim(0.7, 1)
    plt.xlim(-2, SD_train_size)
    plt.grid(linestyle=':')
    plt.savefig(file_out, dpi = 600)
    plt.show()

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

def KNN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test):
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

def tfidf(X_train, X_test, TFIDF_max_features):
    """ Create TF-IDF-matrix of training- and test-features """
    vect = TfidfVectorizer(max_features = TFIDF_max_features, lowercase = False, analyzer = 'word', use_idf = True)

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Convert sparse matrix to array (matrix)
    X_train_dtm = X_train_dtm.toarray()
    X_test_dtm = X_test_dtm.toarray()

    # Store vocabulary (especially important if max_features != None)
    vocabulary = vect.get_feature_names()

    return X_train_dtm, X_test_dtm, vocabulary

###############################################################################
#                               Main loop                                     #
###############################################################################


if tfidf_vec:
    for i in range(n_samples):
        SD_train_size = train_size_vect[i]

        """ This part contains the program (data transformation and ML-algorithms) """


        for iteration in range(iterations):
            # Split data into training- and testset (SciKit)
            kwargs = {
        # https://stackoverflow.com/questions/9539921/how-do-i-create-a-python-function-with-optional-arguments
        # https://realpython.com/python-kwargs-and-args/
                'train_size': SD_train_size,
                'test_size': SD_test_size,
                'random_state': SD_random_state,
                'shuffle': SD_shuffle,
                'stratify': y
                }

            X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs)

            # Create TFIDF-matrix (custom function with SciKit)

            X_train_dtm, X_test_dtm, vocabulary = tfidf(X_train, X_test, TFIDF_max_features)

            # Make prediction (with choosen model)
            if model == 0:
                y_pred_class, y_true_class, train_time, test_time = gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test)

            elif model == 1:
                y_pred_class, y_true_class, train_time, test_time = multinomialNB(X_train_dtm, y_train, X_test_dtm, y_test)

            elif model == 2:
                y_pred_class, y_true_class, test_time = KNN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test)

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
                print("Feature vocabulary: Too many features! to print")

            # Create and show confusion matrix
            conf_matrix = confusion_matrix(y_true_class, y_pred_class)
            #
            #[true neg, false pos]
            #[false neg, true pos]
            #false negative er antallet af spam emails der bliver klassificeret som ikke spam. Det vil vi gerne have til at være så lille som muligt
            #false pos er vigtigere at den er så lille som muligt, da det er ham der              bliver klassificeret som spam

            print("Confusion matrix:")
            print(conf_matrix)

            print('\n')

            # Ekstra statistisk (debugging)
            debugging_accuracy = np.append(debugging_accuracy, accuracy)

        # Print debugging statistiker
        mean = np.mean(debugging_accuracy)

        print("-------- Over all ----------")
        print("Mean: {:.4f}".format(mean))

        output[i] = mean
        #confint[i] = 1.96*np.sqrt((mean*(1-mean))/(SD_train_size))
        #agresti
        p_bar = (mean*SD_test_size+2) / (SD_test_size+4)

        confint[i] = 1.96*np.sqrt((p_bar*(1-p_bar))/(SD_test_size+4))

else:
    for i in range(n_samples):
        SD_train_size = train_size_vect[i]
        for iteration in range(iterations):
            # Split data into training- and testset (SciKit)
            kwargs1 = {                                  # https://stackoverflow.com/questions/9539921/how-do-i-create-a-python-function-with-optional-arguments
                'train_size': SD_train_size,            # https://realpython.com/python-kwargs-and-args/
                'test_size': SD_test_size,
                'random_state': SD_random_state,
                'shuffle': SD_shuffle,
                'stratify': y
                }

            X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs1)

            # Bag of words
            vect = CountVectorizer(lowercase = False, stop_words=None, analyzer = 'word', max_features = BOW_max_features)

            X_train_dtm = vect.fit_transform(X_train)
            X_test_dtm = vect.transform(X_test)

            vocabulary = vect.get_feature_names()

            # Make prediction (with choosen model)
            if model == 1:
                y_pred_class, y_true_class, train_time, test_time = multinomialNB(X_train_dtm, y_train, X_test_dtm, y_test)

            elif model == 2:
                y_pred_class, y_true_class, test_time = KNN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test)

            # Compute accuracy (and show)
            print("Iteration: {}".format(iteration+1))

            numb_test = np.size(y_pred_class)
            numb_correct = np.size(y_pred_class[y_pred_class == y_true_class])

            accuracy = numb_correct/numb_test
            print("Accuracy:  {:.4f}".format(accuracy))

        # =============================================================================
        #     # Show runtime
        #     if model == 1:
        #         print("Runtime:   Training ({:.1f} s.) Testing ({:.1f} s.)".format(train_time, test_time))
        #
        #     else:
        #         print("Runtime:   Testing ({:.1f} s.)".format(test_time))
        # =============================================================================

            # Show features
            if type(BOW_max_features) == int:
                if BOW_max_features <= 15:

                    print("Feature vocabulary:")

                    for word in range(BOW_max_features):
                        print("\t{}. {}".format(word+1,vocabulary[word]))

            else:
                    print("Feature vocabulary: Too many features!")

            # Create and show confusion matrix
            conf_matrix = confusion_matrix(y_true_class, y_pred_class)
            print("Confusion matrix:")
            print(conf_matrix)

            print('\n')

            # Ekstra statistik (debugging)
            debugging_accuracy = np.append(debugging_accuracy, accuracy)

        # Print debugging statistiker
        mean = np.mean(debugging_accuracy)

        print("-------- Over all ----------")
        print("Mean: {:.4f}".format(mean))

        output[i] = mean
        #confint[i] = 1.96*np.sqrt((mean*(1-mean))/(SD_train_size))
        #agresti
        p_tilde = (mean*SD_test_size+2) / (SD_test_size+4)

        confint[i] = 1.96*np.sqrt((p_tilde*(1-p_tilde))/(SD_test_size+4))

if show_plots:
    plot(model_param)

