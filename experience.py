#!/usr/bin/env python
# coding: utf-8


from idea import generator
from encode import clean, encode, flatten
import numpy as np
import requests
import pandas as pd
import string
import nltk

'''Function searches for a random book title related to a given word.'''

def title_search(word):
    
    books = pd.read_html(requests.get(
        'https://www.worldcat.org/search?q=kw%3A'
        + word
        + '&fq=x0%3Abook+%3E+mt%3Afic+%3E+ln%3Aeng&qt=advanced&dblist=638').content)
    
    if len(np.array(books)) != 12:
        return None
    
    book_list = pd.DataFrame(np.array(books)[7])[3]
    titles = list(n.split('  ')[2] for n in book_list)
    
    return titles[int(np.random.rand() * len(titles))]

'''The first half of the data consists of computer-generated strings.''' 

words = list(pd.read_csv('words.csv')['0'])
for i in range(4):
    generated = generator(1000, 100)
    pd.DataFrame(generated).to_csv('generated' + str(i) + '.csv')   

'''The second half is loaded from the internet.'''

titles = []

for i in range(5000):
    
    title = None
    while title == None:
        word = common_words[int(np.random.rand() * len(common_words))]
        title = title_search(word)
        
    titles.append(title)
    
    if (i % 100) == 0:
        print('Iteration: ', i, '\n-----------------\n')
        pd.DataFrame(clean(titles[(i - 100):i])).to_csv('titles' + str(i) + '.csv')
    
titles = np.array(clean(titles))
pd.DataFrame(titles).to_csv('titles.csv')
print(len(titles))

human_text = np.array(pd.read_csv('titles.csv')['0'])
comp_text = np.concatenate(np.array(list(pd.read_csv('generated' + str(i) + '.csv')['0'] for i in range(4))))

c, h = [], []
length = min(len(human_text), len(comp_text))
for i in range(length):
    c.append(int(0))
    h.append(int(1))

data = np.concatenate(
    (np.concatenate(
    (comp_text[:length].reshape(length, 1)
    , np.array(c).reshape(length, 1))
    , axis = 1)
    , np.concatenate(
    (human_text[:length].reshape(length, 1)
    , np.array(h).reshape(length, 1))
    , axis = 1)), axis = 0)

np.random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv('DATA.csv')

