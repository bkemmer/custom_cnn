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

from classifiers.RedeNeural import redeNeuralSoftmax, preditorNeuralSoftmax

from utils import CrossValidacaoEstratificada, Ymulticlasse

def getFilesFolders(path_X):
    p = Path(path_X)
    files = []
    for path in p.rglob('*.npy'):
        files.append(path.parent)
    return np.unique(files)

def runMLP(X_train, y_train, k_folds, k_folds_seed, parametros, folder, classificador_nome, seed, df):
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

        h_size = parametros['h_size']
        ativacao = parametros['ativacao']
        taxa_aprendizado = parametros['taxa_aprendizado']
        max_iteracoes = parametros['max_iteracoes']
      
        W1, b1, W2, b2, custo = redeNeuralSoftmax(X_train_fold, y_train_fold, h_size, ativacao, 
                                                    taxa_aprendizado, max_iteracoes, custo_min=1e-5, seed=seed, plot=False)
        yHat = preditorNeuralSoftmax(X_test_fold, W1, b1, W2, b2, ativacao)
        
        y_test_fold = np.argmax(y_test_fold, axis=1)
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
    features_folder = Path('.', 'Data', 'features_light_MLP')
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
    
    classificador_nome = 'MLP'
    d = datetime.now().strftime('%Y-%m-%d_%H-%M')
    grid_output_folder = Path(path_resultados, '_'.join([classificador_nome, d]))
    grid_output_folder.mkdir(parents=True, exist_ok=True)

    start_time_total = time.time()
    for metodologia in metodologias:

        start_time_metodologia = time.time()
        logging.info('Rodando modelos da metodologia: %s' % metodologia)
        input_folder = Path(features_folder, metodologia)
        
        d = datetime.now().strftime('%Y-%m-%d_%H-%M')
        grid_output = Path(grid_output_folder, '_'.join([metodologia, d]) + '.pickle')
        df = pd.DataFrame(columns=['Preprocessamento', 'Acuracia', 'Classificador', 'Hyperparametros', 'Execution time'])

        param_grid = {
                        'max_iteracoes': [500, 1000, 5000], 
                        'taxa_aprendizado':[0.1, 0.5, 1],
                        'h_size':[10, 100, 500], 
                        'ativacao':['sigmoid','relu'],
                        'cv': 5,
                        'k_folds_seed': 42,
                        'inicializa_rede_seed': 42
                    }

        if test:
            param_grid = {
                        'max_iteracoes': [500], 
                        'taxa_aprendizado':[1],
                        'h_size':[10], 
                        'ativacao':['sigmoid','relu'],
                        'cv': 5,
                        'k_folds_seed': 42,
                        'inicializa_rede_seed': 42
                    }
        k_folds = param_grid['cv']
        k_folds_seed = param_grid['k_folds_seed']
        inicializa_rede_seed = param_grid['inicializa_rede_seed']
        
        y_train = np.load(Path(path_Y, 'y_train.npy'))
        y_train = Ymulticlasse(y_train)
        for folder in getFilesFolders(input_folder):
            logging.info('Lendo input da pasta: %s' % folder)
            X_train = np.load(Path(folder, 'X_train.npy'))

            for max_iteracoes in param_grid['max_iteracoes']:
                for taxa_aprendizado in param_grid['taxa_aprendizado']:
                    for h_size in param_grid['h_size']:
                        for ativacao in param_grid['ativacao']:
                            parametros = { 'h_size': h_size,
                                          'ativacao': ativacao,
                                          'taxa_aprendizado': taxa_aprendizado,
                                          'max_iteracoes': max_iteracoes }
                            df = runMLP(X_train, y_train, k_folds, k_folds_seed, 
                                            parametros, folder, classificador_nome, inicializa_rede_seed, df)
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