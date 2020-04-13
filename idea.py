#!/usr/bin/env python
# coding: utf-8


import nltk
from nltk.corpus import wordnet as wn
from textblob import Word
import numpy as np
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup

words = np.array(pd.read_csv('words.csv')).T[1]
prepositions = np.array(pd.read_csv('prepositions.csv')).T[1]

'''
parts of speech:
    ADJ: a
    ADJ_SAT: s
    ADV: r
    NOUN: n
    VERB: v

wn.synsets('word'): returns all interpretations of the given word

word.definitions: returns all definitions of the given word

wn.syset('word.n.##'): returns the ##-th synset of the word

Word('word'): returns the word as a string

synset._pos: returns part of speech

synset.
    / hypernyms(): returns more general synsets
    / hyponyms(): returns more specific synsets
    / member/part/substance_holonyms(): returns container synsets
    / member/part/substance_meronyms(): returns component synsets
    / similar_tos(): returns synonym synsets
    
synset.lemmas(): returns synonym synsets
synset.lemma_names(): returns synonym strings
synset.lemma_names()[0]: returns the synset as a string

'''

'''Get list of all alphabetic-character-only WordNet words.'''

'''words = [ n for n in wn.all_lemma_names() if n.isalpha() ]
pd.DataFrame(words).to_csv('words.csv')'''

'''Import list of most common prepositions and their frequency.'''

'''preps = np.array(pd.read_html(requests.get(
    'https://www.talkenglish.com/vocabulary/top-50-prepositions.aspx').content)[-1][1])
freq = np.array(pd.read_html(requests.get(
    'https://www.talkenglish.com/vocabulary/top-50-prepositions.aspx').content)[-1][2])

prep_distr = []
for p in range(len(preps)):
    for f in range(freq[p]):
        prep_distr.append(preps[p])
        
prep_distr = np.array(prep_distr)     
np.random.shuffle(prep_distr)
df = pd.DataFrame(prep_distr)
df.to_csv('prepositions.csv')'''

'''Define set of semantic relations based on which the program performs association.'''

relations = [
    'hypernyms'
    , 'hyponyms'
    , 'member_holonyms'
    , 'part_holonyms'
    , 'substance_holonyms'
    , 'member_meronyms'
    , 'substance_meronyms'
    , 'part_meronyms'
    , 'similar_tos'
]

def relate(synset, relation):
    
    if relation == 'hypernyms':
        return synset.hypernyms()
    
    if relation == 'hyponyms':
        return synset.hyponyms()
    
    if relation == 'member_holonyms':
        return synset.member_holonyms()
    
    if relation == 'part_holonyms':
        return synset.part_holonyms()
    
    if relation == 'substance_holonyms':
        return synset.substance_holonyms()
    
    if relation == 'member_meronyms':
        return synset.member_meronyms()
    
    if relation == 'part_meronyms':
        return synset.part_meronyms()
    
    if relation == 'substance_meronyms':
        return synset.substance_meronyms()
    
    if relation == 'similar_tos':
        return synset.similar_tos()
    
    raise ValueError(str(relation) + ': Not a recognized type of semantic relation.')

'''Spreading activation.'''

def spreading(word):
    
    global relations, words
    
    synsets = list(n for n in wn.synsets(word) if n._lemma_names[0] in words)
    
    if len(synsets) == 0:
        return None
    
    synset = synsets[int(np.random.rand() * len(synsets))]
    
    connections = list(
        list(n for n in relate(synset, relation) if (n._lemma_names[0] in words and n._lemma_names[0] != word))
        for relation in relations)
    connections = list(n for n in connections if len(n) != 0)
    
    if len(connections) == 0:
        return None
    
    else:
        connections = np.concatenate(np.array(connections))
        return connections[int(np.random.rand() * len(connections))]._lemma_names[0]

'''Random association function.'''    
        
def associate(word):
        
    global words
    
    levels = min(5, (int(np.random.exponential(scale = 2)) + 1))
        
    for l in range(levels):
        new_word = spreading(word)
        if new_word == None:
            word = words[int(np.random.rand() * len(words))]
        else:
            word = new_word
        
    return word

'''Generate random pieces of text.'''

def generator(pieces, print_iter):
    
    global prepositions, words

    samples = []

    for p in range(pieces):
        
        length = min(10, (2 + int(np.random.exponential(scale = 2))))
        word = words[int(np.random.rand() * len(words))]
        text = word + ' '

        for l in range(length):
            
            if np.random.binomial(n = 1, p = 0.3):
                text += prepositions[int(np.random.rand() * len(prepositions))]
                    
            else:
                if np.random.binomial(n = 1, p = 0.1):
                    text += words[int(np.random.rand() * len(words))]

                else:
                    t = 0
                    while word in text and t < 10:
                        word = associate(word)
                        t += 1
                    if t == 10:
                        text += words[int(np.random.rand() * len(words))]
                    else:
                        text += word
                    
            if l < (length - 1):
                text += ' '
                
        samples.append(text)
        
        if (p % print_iter) == 0:
            print('Iteration: ', p, '\n-----------------\n')
        
    return np.array(samples)

