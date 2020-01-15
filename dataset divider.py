# -*- coding: utf-8 -*- | Wed Jan 15 15:33:37 2020 | Andreas Lau Hansen
import numpy as np
import pandas as pd

df = pd.read_csv('emails.csv')

#all lines with spam / ham
x = np.linspace(0.1, 1, 10)
l = [2736 * x[i] for i in range(10)]
#proportion that is spam
split = 0.1

for i in l:
    spam = df.loc[:i]
    ham = df.loc[1368:(1368+2736-i)]

    out = pd.concat([spam, ham], axis=0)
    stri = 'datasæt/splitemails{:.1f}%spam_size2736.csv'.format(split)
    out.to_csv(stri, encoding='utf-8', index = False)
    split += 0.1





