# -*- coding: utf-8 -*-
""" Program Describtion:
    A program created to be able to test and export results (data) of 
    different ML-algortims under varying circumstances (changing parameters)
    
    Based on the following sources:
    1. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier (KNN)
    2. <add link>
    3. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (TfidfVectorizer)

"""


"""    This part is for changing variables while testing ML-algortihms     """

###############################################################################
###############################################################################
#                      -----  Mission Control  -----                          #
###############################################################################
filename = 'processed_emails_v1.csv'

# Change parameters (split_dataset, tfidfVectorizer and KNN)
SD_random_state = None            # None or int 
SD_shuffle = True 
SD_train_size = 500               # None, int or float 
SD_test_size = 500                # None, int or float 
 
TFIDF_max_features = 3            # None or int 

KNN_k = 3 

# Change which model to run (GaussNB = 1 and KNN = 0)
model = 0

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


def KKN(KNN_k, X_train_dtm, y_train, X_test_dtm, y_test):
    """ Predict classes with K nearest neighbor """
    # Create KNN-model (general)    
    KNN = KNeighborsClassifier(n_neighbors = KNN_k, algorithm = 'brute')
    
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
    kwargs = {                                  # https://stackoverflow.com/questions/9539921/how-do-i-create-a-python-function-with-optional-arguments
        'train_size': SD_train_size,            # https://realpython.com/python-kwargs-and-args/
        'test_size': SD_test_size,
        'random_state': SD_random_state
        }
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs)
    
    # Create TFIDF-matrix (custom function with SciKit)
    X_train_dtm, X_test_dtm, vocabulary = tfidf(X_train, X_test, TFIDF_max_features)
    
    # Make prediction (with choosen model)
    if model == 1:
        y_pred_class, y_true_class, train_time, test_time = gaussianNB(X_train_dtm, y_train, X_test_dtm, y_test)
        
    else:
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
    print("Feature vocabulary:")
    
    for word in range(np.size(vocabulary)):
        print("\t{}. {}".format(word+1,vocabulary[word]))
       
    # Create and show confussion matrix 
    conf_matrix = confusion_matrix(y_true_class, y_pred_class)
    print("Confussion matrix:")
    print(conf_matrix)
    
    print('\n')
    
    # Ekstra statistisk (debugging)
    debbuging_accuracy = np.append(debbuging_accuracy, accuracy)
    
# Print debugging statistiker 
mean = np.mean(debbuging_accuracy)

print("-------- Over all ----------")
print("Mean: {:.4f}".format(mean))
    




###############################################################################
###############################################################################
"""
NB test resultater (debugging):
Overall datasæt med special token substitution: 0.6296, 0.6852, 0.6924, 0.6444
Overall datasæt uden: 0.8392, 0.8380, 0.8400, 0.8520

Spørgsmål:
    1. Hvorfor fanden klarer den sig bedre uden special token substitution?
       Der er forhåbentlig ikke et eller andet build in lort i TFIDFVectorizer
       der tager sig af numre.
"""


"""
Info jeg gerne vil have at den skal spytte ud:
    1. Accuracy (kræver y_pred_class og y_true_class) -|
    2. Runtime -|
    3. Valgte max-features -|
    4. probabilities - ikke den store grund til da disse altid er stort set 0 eller 1 -|
    5. messages classified wrong (og true class) - ikke nødvendig kan ses af confussion matrix -|
    6. confussion matrix -|

"""
    
"""
Noter: Generelt og til NB
1. gnb.score(X_test_dtm,y_test) ikke nødvendig (kan regnes selv)
2. Gauss supporter ikke sparse maricer (i relatiten kunne vi altså godt have kodet i hånden) - se hvordan vi laver om til array med toarray() https://stackoverflow.com/questions/18060232/naive-bayes-probability-always-1
3. Der oplyses at gaussian NB ikke er en god estimator for text-classification (i hvertfald ift. lager, runtime og accuracy) 
4. Der forklares hvorfor vores sandsynligheder er tæt på 0 og 1 ()
5. Der er noget helt galt med vores data eller kode (den genkender ekstremt godt)
"""
    
"""
Noter: KNN
1. Vi kan overveje at undersøge hvad der sker hvis man ændrer parameteren "weights" fra uniform til distance (dvs. jo tættere en nabo er - desto merevægtes den)
2. Har sat parameteren "alorithm" til brute (for brute-force-search) da det er sådan jeg forestiller os vi selv havde regnet det 
3. Hvad er det metric-parameteren dækker over (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) - sørg for at den ikke har inflydelse på resultat

"""
















