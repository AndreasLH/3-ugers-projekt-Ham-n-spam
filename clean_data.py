# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

csv_file = 'emails.csv'
def prepare_data(csv_file):
    """ Function: Loads data and prepares data (feature transformation)
        Input: cvs-file with messages and class (spam = 1 or not spam = 0 )
        Output: pd-DataFrame-object
    """

    # Get raw data and remove unnessecary information from text
    data = pd.read_csv(csv_file)
    data['text'] = data['text'].str.strip('Subject:')
    data['text'] = data['text'].str.strip('fwd :')
    data['text'] = data['text'].str.strip(' re ')

    # Load stop-words, define tokenizer and stemmer
    stop_words = set(stopwords.words('english'))     # https://onlinecoursetutorials.com/nlp/how-to-remove-punctuation-in-python-nltk/
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()

    numb_mails = len(data.text)

    # Remove counter after debugging
    count = 0

    # Loops through each message
    for message in range(numb_mails): # Remove '-5000' after debugging
        # make message lower case
        text = data.text[message].lower()

        # Substitute special tokens with describtive strings
        text = text.replace('$', 'DOLLAR')
        text = text.replace('@', 'EMAILADRESS') # problem ift. " @ "

        # Tokenize + remove punctuation (Problem: @-sign is also removed)
        tokens1 = tokenizer.tokenize(text)

        # Remove stop-words
        tokens2 = [w for w in tokens1 if not w in stop_words]

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

        data.at[message,'text'] = prepared_string
        # Remove after debugging
# =============================================================================
#         if count%50==0:
#             print(count)
#         count += 1
# =============================================================================

    return data

result = prepare_data('emails.csv')
print(result.text[0])

result.to_csv('processed_emails.csv', encoding='utf-8', index = False)


"""
Problems encountered so far:
    1. Ord bliver også splittet op ved f.eks. '-tegn (tror jeg) - dette
       betyder f.eks. at can't forsvinder
    2. Vores datasæt indeholder en hel masse i'er i stedet for l'er og omvendt
    3. Problem ift. " @ "-mellemrum ved emailadresse. Find ud af om dette er
       en regel, og jeg i så fald kan inkorperere det at fjerne det omkring.
       Problematisk eksempel: '75 @ tfi . kpn . com'
    4. Lav evt. datoer til speciel token med navn
    5. Der antages at hver gang der står et nummer som f.eks. "555 55" efter
       at der er blevet fjeret punktumer og kommaer - så er der tale om et tal
       og ikke flere tal som i sætningen "Den koster kr. 55. 65 mennersker har..."
    6. Forskel på dollar i tekst og "DOLLAR" assignment
    7. "g" and "q"



"""






