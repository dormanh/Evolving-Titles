#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from idea import generator, associate
from encode import encode, flatten, clean
import pandas as pd
from nltk.corpus import wordnet as wn
import matplotlib.pyplot

model = pickle.load(open('evaluator_tree.sav', 'rb'))

'''Evolution parameters.'''

population_size = 1000
generations = 500
min_fitness = 0.5
copy_prop = 0.1
mutation_size = 0.5

'''Length for encoding strings.'''
length = max(list(len(t) for t in np.array(pd.read_csv('DATA.csv')['0'])))

generation_zero = np.array(pd.read_csv('generation_zero.csv')['0'])

'''Give birth to generation zero.'''

generation_zero = generator(population_size, 100)
pd.DataFrame(generation_zero).to_csv('generation_zero.csv')

words = np.array(pd.read_csv('words.csv')).T[1]
prepositions = np.array(pd.read_csv('prepositions.csv')).T[1]
all_words = np.concatenate((words, prepositions))

'''Evaluate fitness of specimens.'''

def evaluate(generation):
    
    global model, min_fitness, length
    
    fitness = model.predict(encode(generation, length))
    avg_score = fitness.mean()
    survivors = fitness.argsort()[-int(len(generation) * min_fitness):][::-1]
    creeps = fitness.argsort()[:int(len(generation) * min_fitness)][::-1]
    
    return np.array([survivors, creeps, avg_score])

'''Sort fittest specimens.'''

def sort_next_gen(prev_gen, survivors):
    
    next_gen = []
    
    for s in survivors:
        next_gen.append(prev_gen[s])
        
    return np.array(next_gen)

'''Generate new samples by mutating some specimens.'''

def mutate(specimens):
    
    global mutation_size, all_words
    
    mutations = []
        
    for spec in specimens:
        elements = spec.split(' ')
        change, copy = train_test_split(
            elements, train_size = mutation_size, test_size = 1 - mutation_size)
        new = []
        for w in change:
            if wn.synsets(w) != []:
                new.append(associate(w))
            else:
                new.append(all_words[int(np.random.rand() * len(all_words))])
        diff_order = np.array(elements)
        np.random.shuffle(diff_order)
        mutation1 = ''
        mutation2 = ''
        for e in range(len(elements)):
            mutation2 += diff_order[e]
            if elements[e] in change:
                mutation1 += new[change.index(elements[e])]
            else:
                mutation1 += elements[e]
            if e < (len(elements) - 1):
                mutation1 += ' '
                mutation2 += ' '
                    
        mutations.append(mutation1)
        mutations.append(mutation2)
        
    return np.array(mutations)

'''Generate new samples by crossing over some specimens.

def crossover(specimens):
    
    children = []
    
    for i in range(2):
    
        singles = list(specimens)

        while len(singles) > 1:

            pair1 = singles[int(len(singles) * np.random.rand())]
            singles.remove(pair1)
            pair2 = singles[int(len(singles) * np.random.rand())]
            singles.remove(pair2)
            elements1 = pair1.split(' ')
            elements2 = pair2.split(' ')
            child1 = ''
            child2 = ''
            for e in range(int(len(elements1) / 2)):
                child1 += elements1[e] + ' '
            for e in range(int(len(elements2) / 2)):
                child2 += elements2[e] + ' '
            for e in range(int(len(elements1) / 2), len(elements1)):
                child2 += elements1[e]
                if e < (len(elements1) - 1):
                    child2 += ' '
            for e in range(int(len(elements2) / 2), len(elements2)):
                child1 += elements2[e]
                if e < (len(elements2) - 1):
                    child1 += ' '
            children.append(child1)
            children.append(child2)

        if len(singles) == 1:
            children.append(singles[0])
        
    return np.array(children)'''

'''Perform evolution process.'''

generation = generation_zero
scores = []

for g in range(generations):
    
    evaluated = evaluate(generation)
    scores.append(evaluated[2])
    next_gen_material = sort_next_gen(generation, evaluated[0])
    creeps = sort_next_gen(generation, evaluated[1])
    np.random.shuffle(creeps)
    creeps = creeps[:int(population_size * min_fitness * copy_prop)]
    copied, mutating = train_test_split(next_gen_material, train_size = copy_prop, test_size = 1 - copy_prop)
    
    mutated = mutate(mutating)
    
    generation = np.concatenate((copied, creeps, mutated))
    np.random.shuffle(generation)
    
    if (g % 10) == 0:
        print('Generation: ', g)
        print('Some members: ', generation[:5])
        print('-------------------------------------\n')
    if (g % 50) == 0:
        pd.DataFrame(generation).to_csv('evolving_tree' + str(g) + '.csv')
    
print(generation)
df = pd.DataFrame(generation)
df.to_csv('final_generation_tree.csv')

