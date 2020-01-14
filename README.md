# Links til forståelse af Naive Bayes klassifikation

I opgaven har vi indtil videre kun brugt Naive Bayes i Gauss versionen (kontinuerte værdier). Jeg kan dog anbefale at læse de andre links igennem for at få en forståelse. Det anbefales dog at få en forståelse for alle 3 versioner (da det let vil kunne impelmenteres som næste del af projektet).

## Vigtigste links til Naive Bayes:
1.	https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/ (generel gennemgang af Naiv Bayes)
2.	https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/ (specifik gennemgang af gauss-eksempel kode uden libraries – anbefalet at læse generelle gennemgang først)
3.	https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes (SciKit beskrivelse)
4.	https://sebastianraschka.com/Articles/2014_naive_bayes_1.html (generel gennemgang af Naive Bayes ift. Gauss, multinomial og bernoulli)
5.	https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67 (kode eksempel til multinomial – dog virker det som om det vil være bedst at arbejde videre på kode eksempålet til Gauss hvis denne type implementering skulle forettages)

## Dataset
https://www.kaggle.com/karthickveerakumar/spam-filter/version/1

## filer
Hovedfil er `Samlet_test_kode.py`

## Versioner af datasæt
Konstante parametere i datatransformation:
1.	Fjern email-informations tags (fw, re, subject, cc)
2.	Erstat specielle tegn eller tokens med strings (NUMBER, DOLLAR, EMAILADRESS, URL)
3.	Lav alt lower case
4.	Fjern punctuation 
5.	Tag stammen af alle tokens (stemming)

Overblik over datasæt versioner:

Version 1: Fjern stop-words (standard-sæt)

Version 2: Fjern få enron-specifikke ting  - Fjern enron, ect, com og kaminski (kunne fjerne mange flere men giver ikke mening ifølge Mikkel)

Version 3: Alle de konstante parametre (se ovenfor)
