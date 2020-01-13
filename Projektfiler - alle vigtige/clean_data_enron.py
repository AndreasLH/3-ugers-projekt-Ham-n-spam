# -*- coding: utf-8 -*-

""" This code is meant to be used to clean a csv-dataset and make the necessary 
    normalization (punctuation, removing, stopwords etc.). Finally it will
    export the clean dataset as a csv-file.
    
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
    stop_words = set(stopwords.words('english'))     # https://onlinecoursetutorials.com/nlp/how-to-remove-punctuation-in-python-nltk/
    tokenizer = RegexpTokenizer(r'\w+')              # https://riptutorial.com/nltk/example/27285/filtering-out-stop-words
    ps = PorterStemmer()                             # https://www.geeksforgeeks.org/python-stemming-words-with-nltk/

    numb_mails = len(data.text)  

    # Loops through each message (email)
    for message in range(numb_mails):
        # make message lower case
        text = data.text[message]
        
        # Substitute special tokens with describtive strings (before removed ny tokenizer)
        text = text.replace('$', 'DOLLAR')
        text = text.replace('@', 'EMAILADRESS')
        text = text.replace('https', 'URL')
        text = text.replace('www', 'URL')
        
        # Remove unescessary information 
        text = text.replace('Subject', '')
        text = text.replace('cc', '')
        
# =============================================================================
#         text = text.replace('enron', '')
#         text = text.replace('ect', '')
#         text = text.replace(' com ', '')
#         text = text.replace('kaminski', '')
# =============================================================================
        
        # Make text lower case
        text = text.lower()
        
        # Tokenize + remove punctuation 
        tokens2 = tokenizer.tokenize(text)        
        
# =============================================================================
#         # Remove stop-words 
#         tokens2 = [w for w in tokens1 if not w in stop_words] #https://riptutorial.com/nltk/example/27285/filtering-out-stop-words
#         
# =============================================================================
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
        
        data.at[message,'text'] = prepared_string  # https://stackoverflow.com/a/13842286/12428216

    return data 
        
result = clean_data('Datasæt\emails.csv')
result.to_csv('processed_emails_v3.csv', encoding='utf-8', index = False)








# Debuggning-part (remove later)
# =============================================================================
# a = result.text[60]
# print(a.find('vinc'))
# print(len(a[2000,:]))
# print(data.text[60][2000:])
#
# print(result.text)
# =============================================================================


# Information om scriptet (ift. flaws)
# ============================================================================= 
# Problems encountered so far:
#     1. Ord bliver også splittet op ved f.eks. '-tegn (tror jeg) - dette 
#        betyder f.eks. at can't forsvinder 
#     2. Vores datasæt indeholder en hel masse i'er i stedet for l'er og omvendt.
#        Det samme gælder ift. "q" og "g". Nedsætter formentlig vores algoritmers
#        effektivitet. 
#     3. Bemærk at ikke hele mailadresse fjernes - kun @ udskriftes med string. 
#     4. Der antages at hver gang der står et nummer som f.eks. "555 55" efter
#        at der er blevet fjeret punktumer og kommaer - så er der tale om et tal
#        og ikke flere tal som i sætningen "Den koster kr. 55. 65 mennersker har..."
# =============================================================================

    
    
  
# Hvad med ". com "







