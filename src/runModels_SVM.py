#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Lê os arquivos binários já pré-processados e executa os classificadores

Usage:

    python3 runModels_neg_corr.py

Returns:
    
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd

# from classifiers.svm import svm
from classifiers.svm2 import SVM
from classifiers.kernels import kernel_linear, kernel_polinomial, kernel_rbf

from utils import CrossValidacaoEstratificada, Ymulticlasse

def getFilesFolders(path_X):
    p = Path(path_X)
    files = []
    for path in p.rglob('*.npy'):
        files.append(path.parent)
    return np.unique(files)

def runSVM(X_train, y_train, k_folds, k_folds_seed, parametros, folder, classificador_nome, df):
    start_time = time.time()
    # executa a cross validação estratificada
    X_train_folds = CrossValidacaoEstratificada(X_train, y_train, k_folds, k_folds_seed)
    
    resultados_fold = np.zeros(k_folds)
    for k_fold, fold in enumerate(X_train_folds):
        # Separando por fold
        others_ids = set(range(len(X_train))).difference(set(fold))
        others_ids = list(others_ids)

        X_train_fold = X_train[fold]
        y_train_fold = y_train[fold]
        X_test_fold = X_train[others_ids]
        y_test_fold = y_train[others_ids]

        kernel_type = parametros['kernel_type']
        degree = parametros['degree']
        C = parametros['C']

        # v1
        # svm_instance = svm()
        # svm_instance.fit(X_train, y_train, kernel_type, degree, C)
        # yHat = svm_instance.predict(X_test_fold)

        #v2
        if kernel_type == 'linear':
            svm_clf = SVM(kernel_type, kernel=kernel_linear, C=C)
        elif kernel_type == 'polynomial':
            svm_clf = SVM(kernel_type, kernel=kernel_polinomial, grau=degree, escalar=1, C=C)
        # else:
        #     svm_clf = SVM(kernel=kernel_rbf, gamma=gamma)
        svm_clf.fit(X_train_fold, y_train_fold)
        yHat = svm_clf.predict(X_test_fold)

        resultados_fold[k_fold] = (yHat==y_test_fold).sum()/len(yHat)

    acc = resultados_fold.mean()

    logging.info('%s: acc: %2f params: %s' % (folder, acc, str(parametros)))
    df_run = pd.DataFrame({'Preprocessamento': folder,
                            'Acuracia': acc,
                            'Classificador': classificador_nome,
                            'Hyperparametros': str(parametros),
                            'Execution time': '%.1f' % (time.time() - start_time)
                            }, index=[0])
    return pd.concat([df, df_run])

def main(args):

    # Configurações do grid
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    #path com as features
    features_folder = Path('.', 'Data', 'features_light')
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

    start_time_total = time.time()
    for metodologia in metodologias:

        start_time_metodologia = time.time()
        logging.info('Rodando modelos da metodologia: %s' % metodologia)
        input_folder = Path('.', 'Data', 'features_light', metodologia)
        
        d = datetime.now().strftime('%Y-%m-%d_%H-%M')
        classificador_nome = 'SVM' 
        grid_output = Path(path_resultados, '_'.join([classificador_nome, metodologia, d]) + '.pickle')
        df = pd.DataFrame(columns=['Preprocessamento', 'Acuracia', 'Classificador', 'Hyperparametros', 'Execution time'])


        param_grid = {
                        'C': [1, 10, 100],
                        'degree': [1, 2, 3],
                        'kernel': ['linear', 'polynomial'],
                        'cv': 5,
                        'k_folds_seed': 42,
                    }
        if test:
            param_grid = {'C': [1], 'degree': [1], 'kernel': ['linear'],
                            'cv': 5, 'k_folds_seed': 42}

        k_folds = param_grid['cv']
        k_folds_seed = param_grid['k_folds_seed']

        y_train = np.load(Path(path_Y, 'y_train.npy'))
        y_train[y_train == 0] = -1
        
        for folder in getFilesFolders(input_folder):
            logging.info('Lendo input da pasta: %s' % folder)
            X_train = np.load(Path(folder, 'X_train.npy'))

            for C in param_grid['C']:
                for kernel_type in param_grid['kernel']:
                    if kernel_type == 'polynomial':
                        for degree in param_grid['degree']:
                            parametros = { 'kernel_type': kernel_type,
                                          'degree': degree,
                                          'C': C }
                            df = runSVM(X_train, y_train, k_folds, k_folds_seed, 
                                        parametros, folder, classificador_nome, df)
                    elif kernel_type == 'linear':
                        parametros = { 'kernel_type': kernel_type,
                                          'degree': None,
                                          'C': C }
                        df = runSVM(X_train, y_train, k_folds, k_folds_seed, 
                                        parametros, folder, classificador_nome, df)
                    df.to_pickle(grid_output)

        logging.info('Execução da metodologia %s: %.1f segundos.' % (metodologia, 
                                                                      time.time() - start_time_metodologia))
    logging.info('Execução total: %.1f segundos.' % (time.time() - start_time_total))
    
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