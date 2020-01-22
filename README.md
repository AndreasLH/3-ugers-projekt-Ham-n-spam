# Filer
Alle hovedeksperimentfiler er ligger i root folder. Derudover er de scripts der er brugt til at lave plots i mappen `Plotscripts`

Hovedfil er `Samlet_test_kode.py`

Yderligere filer er

- `Samlet_test_kode - TFIDF.py` - Test scriptet til de modeller der kører TF-IDF
- `Samlet_test_kode-Bag-of-words.py`- Test scriptet til de modeller der kører Bag of words
- `clean_data_enron.py` - data cleaning scriptet
- `split5050_alternativ.py` - data split scriptet


## Dataset
Vores datasæt er fra
https://www.kaggle.com/karthickveerakumar/spam-filter/version/1

### Versioner af datasæt
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
