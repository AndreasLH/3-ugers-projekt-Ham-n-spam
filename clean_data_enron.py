# -*- coding: utf-8 -*-

""" This code is meant to be used to clean a csv-dataset and make the
necessary normalization (punctuation, removing, stopwords etc.).
Finally it will export the clean dataset as a csv-file.

    Link to dataset:
    https://www.kaggle.com/karthickveerakumar/spam-filter/version/1
"""

# Import libraries
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_data(csv_file):
    """ Function: Loads data and prepares data (feature transformation)
        Input: cvs-file with messages and class (spam = 1 or not spam = 0 )
        Output: pd-DataFrame-object
    """

    # Get raw data and remove unnessecary email-related tags from text
    data = pd.read_csv(csv_file)

    data['text'] = data['text'].str.strip('fw :')
    data['text'] = data['text'].str.strip(' re : ')

    # Load stop-words, define tokenizer and stemmer
#onlinecoursetutorials.com/nlp/how-to-remove-punctuation-in-python-nltk/
# https://riptutorial.com/nltk/example/27285/filtering-out-stop-words
# https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()

    numb_mails = len(data.text)

    # Loops through each message (email)
    for message in range(numb_mails):
        # make message lower case
        text = data.text[message]

# Substitute special tokens with descriptive strings
#(before removed ny tokenizer)
        text = text.replace('$', 'DOLLAR')
        text = text.replace('@', 'EMAILADRESS')
        text = text.replace('https', 'URL')
        text = text.replace('www', 'URL')

        # Remove unescessary information
        text = text.replace('Subject', '')
        text = text.replace('cc', '')

        # Make text lower case
        text = text.lower()

        # Tokenize + remove punctuation
        tokens1 = tokenizer.tokenize(text)

        # Remove stop-words
        tokens2 = [w for w in tokens1 if not w in stop_words]
        #https://riptutorial.com/nltk/example/27285/filtering-out-stop-words

        # Stemming tokens
        numb_tokens = len(tokens2)

        for token in range(numb_tokens):
            tokens2[token] = ps.stem(tokens2[token])

        # Sustitute number (special token) with 'NUMBER'
        #(numbers can be split by with space)
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
        # https://stackoverflow.com/a/13842286/12428216

    return data

result = clean_data('Datasæt\emails.csv')
result.to_csv('processed_emails.csv', encoding='utf-8', index = False)

print(len(result.text))
