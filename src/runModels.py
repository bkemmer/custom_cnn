#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Lê os arquivos binários já pré-processados e executa os classificadores

Usage:

    python3 runModels.py 

Returns:
    
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def getFilesFolders(path_X):
    p = Path(path_X)
    files = []
    for path in p.rglob('*.npy'):
        files.append(path.parent)
    return np.unique(files)

def runModels(path_X, path_Y, classifier, param_grid, n_jobs=-1):
    grid_searchs = {}
    grid_search_best_estimators = {}

    folders = getFilesFolders(path_X)
    y_train = np.load(Path(path_Y, 'y_train.npy'))
    # y_test = np.load(Path(path_Y, 'y_test.npy'))
    
    for folder in folders:
        logging.info('Lendo input da pasta: %s' % folder)
        X_train = np.load(Path(folder, 'X_train.npy'))
        # X_test = np.load(Path(folder, 'X_test.npy'))

        grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring="accuracy", n_jobs=n_jobs, param_grid=param_grid, verbose=10)
        grid_search.fit(X_train, y_train)
    
        grid_searchs[str(folder)] = grid_search.cv_results_
        grid_search_best_estimators[str(folder)] = grid_search.best_estimator_
    
    return grid_searchs, grid_search_best_estimators

def runGridSearch(input_folder, path_Y, Classificador_nome, classifier, param_grid, grid_output, df):
    grid_searchs, grid_search_best_estimators = runModels(input_folder, path_Y, classifier, param_grid)
    
    for k,grid_search in grid_searchs.items():
        for mean_test_score, params in zip(grid_search['mean_test_score'], grid_search['params']):
            acc = np.squeeze(mean_test_score)
            logging.info('%s: acc: %2f param: %s' % (k, acc, params))
            df_run = pd.DataFrame({'Preprocessamento': k,
                                    'Acuracia': acc,
                                    'Classificador': Classificador_nome,
                                    'Hyperparametros': str(params)
                                    }, index=[0])
            df = pd.concat([df, df_run])
        df.to_pickle(grid_output)
    return df

def main(args):
    
    if args.metodologia is not None:
        metodologias = [args.metodologia.strip()]
    else:
        metodologias = ['VJ_resize_Concat_Flatten_ICA', 'VJ_resize_Concat_HOG_PCA', 
                        'VJ_resize_HOG_Concat_Normalize', 'VJ_resize_HOG_Concat_PCA',
                        'VJ_resize_HOG_Concat_PCA_normalize', 'VJ_resize_LBP', 'VJ_resize_LBPH']

    for metodologia in metodologias:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logging.info('Rodando modelos da metodologia: %s' % metodologia)

        input_folder = Path('.', 'Data', 'features', metodologia)
        path_Y = Path('.', 'Data', 'Processados')
        
        d = datetime.now().strftime('%Y-%m-%d_%H-%M')
        path_resultados = Path('.', 'Data', 'Processados', 'Resultados')
        path_resultados.mkdir(parents=True, exist_ok=True)

        grid_output = Path(path_resultados, metodologia + '_' + d + '.pickle')
        df = pd.DataFrame(columns=['Preprocessamento', 'Acuracia', 'Classificador', 'Hyperparametros'])
        
        # SVC
        param_grid = [{'C': [0.1, 1, 5, 10, 100], 'degree': [1,2,3,4], 'kernel': ['poly', 'rbf']}]
        # param_grid = [{'C': [1], 'degree': [1], 'kernel': ['poly','rbf']}]
        Classificador_nome = 'SVM'
        classifier = SVC()
        df = runGridSearch(input_folder, path_Y, Classificador_nome, classifier, param_grid, grid_output, df)
        
        param_grid = {  'solver': ['sgd'], 
                        'max_iter': [1000, 2000, 5000], 
                        'learning_rate':['constant'], 
                        'learning_rate_init':[0.1, 1, 10],
                        'hidden_layer_sizes':[10, 100, 1000], 
                        'activation':['logistic','relu']}

        # param_grid = {  'solver': ['sgd'], 
        #                 'max_iter': [1000, ], 
        #                 'learning_rate':['constant'], 
        #                 'learning_rate_init':[1],
        #                 'hidden_layer_sizes':[10], 
        #                 'activation':['logistic','relu']}

        Classificador_nome = 'MLP'
        classifier = MLPClassifier()
        df = runGridSearch(input_folder, path_Y, Classificador_nome, classifier, param_grid, grid_output, df)

    a=1
if __name__ == '__main__':

    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("-m", "--metodologia", required=False,  
                        help="Metodologia a ser executada")
    
    args = ap.parse_args()

    main(args)