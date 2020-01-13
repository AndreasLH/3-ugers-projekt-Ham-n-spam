# -*- coding: utf-8 -*- | Fri Jan 10 15:02:22 2020 | Andreas Lau Hansen

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')

df = pd.read_csv('emails.csv')
print(f'consists of {df.shape[0]} rows and {df.shape[1]} columns')
print(df.head())

def lexical_diversity(text):
    return len(set(text)) / len(text)

df['Lexical diversity'] = df['text'].apply(lexical_diversity)
df['Length'] = df['text'].apply(len)

def plot_feature(feature):
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    for i, c in enumerate([0, 1]):
        axs[i].hist(df[df['spam'] == c][feature], bins = 50)
        axs[i].set_ylabel('Count')
        axs[i].set_xlabel(feature)
        axs[i].set_title(f'{feature} of {c}')

plot_feature('Lexical diversity')
plot_feature('Length')
plt.show()