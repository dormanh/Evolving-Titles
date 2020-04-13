#!/usr/bin/env python
# coding: utf-8


import numpy as np
import string
import pandas as pd

def encode(array, length):
        
    alphabet = string.ascii_lowercase + ' '
    alpha = np.array([i for i in alphabet])
    codes = np.zeros(shape = (len(array), length))#, len(alphabet)))
    
    for n in range(len(array)):
        for l in range(length):
            if l < len(array[n]):
                codes[n][l] =np.where(alpha == array[n][l])[0][0]# += 1
        
    return codes

def flatten(array):

    flat = []
    for n in array:
        flat.append(n.flatten())

    return np.array(flat)

def clean(text_list):
    
    alphabet = string.ascii_lowercase + ' '
    clean_text = []
    
    for t in text_list:
        if sum(list((char.lower() in alphabet) for char in t)) == len(t):
            clean_text.append(t.lower())
        
    return np.array(clean_text)

