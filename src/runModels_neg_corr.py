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


from classifiers.ensemble_negative_correlation import ensemble

from utils import CrossValidacaoEstratificada

from itertools import product

def my_product(dicionario):
    return (dict(zip(dicionario.keys(), values)) for values in product(*dicionario.values()))

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

        # grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring="accuracy", n_jobs=n_jobs, 
        #                             param_grid=param_grid, verbose=10)
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

    # Configurações do grid
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    #path com as features
    features_folder = Path('.', 'Data', 'features')
    # path onde está as variáveis dependentes Y
    path_Y = Path('.', 'Data', 'Processados')
    # path onde serão salvos os resultados .pickle
    path_resultados = Path('.', 'Data', 'Processados', 'Resultados')
    path_resultados.mkdir(parents=True, exist_ok=True)
    
    # leitura dos argumentos de entrada
    test = args.test
    if args.metodologia is not None:
        metodologias = [args.metodologia.strip()]
    elif args.all:
        metodologias = [f.name for f in features_folder.iterdir() if f.is_dir()]
    # caso seja feita uma execução de somente algumas features
    else:
        metodologias = ['VJ_resize_Concat_Flatten_ICA', 'VJ_resize_Concat_HOG_PCA', 
                        'VJ_resize_HOG_Concat_Normalize', 'VJ_resize_HOG_Concat_PCA',
                        'VJ_resize_HOG_Concat_PCA_normalize', 'VJ_resize_LBP', 'VJ_resize_LBPH']


    for metodologia in metodologias:

        logging.info('Rodando modelos da metodologia: %s' % metodologia)
        input_folder = Path('.', 'Data', 'features', metodologia)
        
        d = datetime.now().strftime('%Y-%m-%d_%H-%M')

        grid_output = Path(path_resultados, metodologia + '_' + d + '.pickle')
        df = pd.DataFrame(columns=['Preprocessamento', 'Acuracia', 'Classificador', 'Hyperparametros'])

        param_grid = {
                        'n_max': [1000, 2000, 5000, 10000], 
                        'alfa': [0.1, 1, 5, 10],
                        'lamdba': [0, 0.25, 0.5, 0.75, 1],
                        'num_hidden': [10, 100, 1000],
                        'number_classifiers': [2, 3, 4, 5],
                        'cv': 3,
                        'k_folds_seed': 42
                    }
        if test:
            param_grid = {
                            'n_max': [1000], 
                            'alfa': [1],
                            'lamdba': [0.5],
                            'num_hidden':[100],
                            'number_classifiers': [3],
                            'cv': 3,
                            'k_folds_seed': 42
                        }

        k_folds = param_grid['cv']
        k_folds_seed = param_grid['k_folds_seed']

        y_train = np.load(Path(path_Y, 'y_train.npy'))

        Classificador_nome = 'Negative Correlation'
        for folders in getFilesFolders(input_folder):
            logging.info('Lendo input da pasta: %s' % folder)
            X_train = np.load(Path(folder, 'X_train.npy'))

            
            for num_hidden in param_grid['num_hidden']:
                for number_classifiers in param_grid['number_classifiers']:
                    for n_max in param_grid['n_max']:
                        for alfa in param_grid['alfa']:
                            for lamdba in param_grid['lamdba']:
                                
                                classifier = ensemble(num_hidden = num_hidden, number_classifiers = number_classifiers)

                                # executa a cross validação estratificada
                                X_train_folds = CrossValidacaoEstratificada(X_train, y_train, k_folds, k_folds_seed)

                                for k_fold, fold in enumerate(X_train_folds):
                                    # Separando por fold
                                    others_ids = set(range(len(v))).difference(set(fold))
                                    others_ids = list(others_ids)

                                    X_train_fold = X_train[fold]
                                    y_train_fold = y_train[fold]
                                    X_test_fold = X_train[others_ids]
                                    y_test_fold = y_train[others_ids]

                                    classifier.fit(X=X_train_fold, y_d=y_train_fold, lamdba=lamdba, alfa=alfa, n_max=n_max)
                                    yHat = classifier.predict(X_test_fold)






            
    

            

    # # ajuda
    # df = runGridSearch(input_folder, path_Y, Classificador_nome, classifier, param_grid, grid_output, df)


    # df_folds, df_folds_agrupado, df_topN = treinarModelos(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, 
    #                                                         topN=3, parametros=parametros, save_pickle=True, save_excel=True)

    # df_folds = treinaCrossValidacao(nome_dataset, X_treino, y_treino, modelos, parametros=parametros)
    # df_folds_agrupado = agrupa_kfolds(df_folds)




    # folders = getFilesFolders(path_X)
    # y_train = np.load(Path(path_Y, 'y_train.npy'))
    # # y_test = np.load(Path(path_Y, 'y_test.npy'))
    
    # for folder in folders:
    #     logging.info('Lendo input da pasta: %s' % folder)
    #     X_train = np.load(Path(folder, 'X_train.npy'))
    #     # X_test = np.load(Path(folder, 'X_test.npy'))

    #     # grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring="accuracy", n_jobs=n_jobs, 
    #     #                             param_grid=param_grid, verbose=10)
    #     grid_search.fit(X_train, y_train)
    
    #     grid_searchs[str(folder)] = grid_search.cv_results_
    #     grid_search_best_estimators[str(folder)] = grid_search.best_estimator_

    # a=1

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("-m", "--metodologia", required=False,  
                        help="Metodologia a ser executada")
    ap.add_argument("-all", "--all", action="store_true", default=False, 
                        help="Flag para rodar todos os modelos que estão na pasta de features")
    ap.add_argument("-test", "--test", action="store_true", default=False, 
                        help="Flag para rodar apenas um modelo de teste")
    args = ap.parse_args()

    main(args)