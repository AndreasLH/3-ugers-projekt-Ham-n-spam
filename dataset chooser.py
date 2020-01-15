# -*- coding: utf-8 -*- | Wed Jan 15 15:33:37 2020 | Andreas Lau Hansen
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()

df = pd.read_csv('emails.csv')

#all lines with spam
df[df['spam'] == 1]
#proportion that is spam
len(df[df['spam'] == 1]) / len(df[df['spam'] == 0])

sns.pairplot(df, hue='text');
