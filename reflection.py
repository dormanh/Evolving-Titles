#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pickle
from encode import encode

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

pd.set_option('display.max_rows', 500)

data = pd.read_csv('DATA.csv')
X = np.array(data['0'])
Y = np.array(data['1'])
length = max(list(len(t) for t in X))
X = encode(X, length)

models = [
    LinearRegression,
    LogisticRegression,
    MLPRegressor,
    Lasso,
    ElasticNet,
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor
]

grids = {
    'LinearRegression':{},
    'LogisticRegression': {'max_iter': [10000]},
    'MLPRegressor':{'hidden_layer_sizes':[(6, 7), (7, 6), (5, 6), (6, 5), (6, 8), (8, 7)], 'max_iter':[10000]},
    'Lasso':{'alpha':[10 ** n for n in [-10, -5, -3, -1, 0, 1]]},
    'ElasticNet':{'alpha':[10 ** n for n in [-10, -5, -3, -1, 0, 1]]},
    'KNeighborsRegressor':{'weights':['uniform', 'distance'], 'n_neighbors':[3, 5, 8, 10, 20, 30, 100],
                           'n_jobs':[mp.cpu_count()]},
    'DecisionTreeRegressor':{'max_depth':[3, 4, 5, 6, 7, 8]},
    'RandomForestRegressor':{'max_depth':[4, 5, 6], 'n_estimators':[10, 50, 100, 200], 'n_jobs':[mp.cpu_count()]}
}

performance_df = pd.DataFrame()
cvinfo = ['mean_fit_time', 'mean_train_score', 'mean_test_score', 'std_test_score', 'params']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
best_params = {}

for model in models:
    
    modelname = model.__name__
    print("\n\n\n-----------\n-----------\nModel name: ", modelname)
    
    par_grid = grids[modelname]
    grid_search = GridSearchCV(model(), par_grid, cv = 8, verbose = 1, return_train_score = True, n_jobs = mp.cpu_count()) 
    
    grid_search.fit(X_train, Y_train)

    print("Test set score: {:.2f}".format(grid_search.score(X_test, Y_test)))
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("Best estimator:\n{}".format(grid_search.best_estimator_))
    
    best_params[modelname] = grid_search.best_params_
    
    performance_df = pd.concat([performance_df,
        pd.DataFrame({**{c: grid_search.cv_results_[c] for c in cvinfo},'name': modelname})])

model = RandomForestRegressor(max_depth = 6, n_jobs = 2, n_estimators = 200)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, test_size = 0.1)

model.fit(X_train, Y_train)

print('Train score: ', model.score(X_train, Y_train))
print('Test score: ', model.score(X_test, Y_test))

pickle.dump(model, open('evaluator_tree.sav', 'wb'))

