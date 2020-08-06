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

from classifiers.RedeNeuralNegEnsemble import redeNeuralSoftmaxEnsemble, preditorNeuralSoftmaxEnsemble

from utils import CrossValidacaoEstratificada, Ymulticlasse


def getFilesFolders(path_X):
    p = Path(path_X)
    files = []
    for path in p.rglob('*.npy'):
        files.append(path.parent)
    return np.unique(files)

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
    
    # Manualmente não rodar alguma metodologia
    skipped = ['VJ_resize_LBPH_normalized', 'VJ_resize_HOG_Concat_Normalize']
    metodologias = [m for m in metodologias if m not in skipped]

    classificador_nome = 'negative_correlation'
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
                        'n_max': [500, 1000],#, 10000], 
                        'alfa': [0.1, 0.5, 1],
                        'lamdba': [0.5, 1],
                        'num_hidden': [10, 100],#, 1000],
                        'number_classifiers': [2, 3],
                        'cv': 5,
                        'k_folds_seed': 42,
                        'inicializa_rede_seed': 42
                    }
        if test:
            param_grid = {
                            'n_max': [1000], 
                            'alfa': [1],
                            'lamdba': [0.5],
                            'num_hidden':[100],
                            'number_classifiers': [3],
                            'cv': 3,
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

            for num_hidden in param_grid['num_hidden']:
                for number_classifiers in param_grid['number_classifiers']:
                    for n_max in param_grid['n_max']:
                        for alfa in param_grid['alfa']:
                            for lamdba in param_grid['lamdba']:
                                
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

                                    parametros, custo = redeNeuralSoftmaxEnsemble(X_train_fold, y_train_fold, n_redes=number_classifiers, lamdba=lamdba, 
                                                                                    h_size=num_hidden, ativacao='sigmoid', taxa_aprendizado=alfa,
                                                                                    max_iteracoes=n_max, custo_min=1e-7, 
                                                                                    seed=inicializa_rede_seed,
                                                                                    plot=False)
                                    yHat = preditorNeuralSoftmaxEnsemble(X_test_fold, parametros, ativacao='sigmoid')
                                    yHat = Ymulticlasse(yHat)

                                    resultados_fold[k_fold] = np.all(yHat==y_test_fold, axis=1).sum()/len(yHat)

                                acc = resultados_fold.mean()
                                
                                params = {  'num_hidden': num_hidden,
                                            'number_classifiers': number_classifiers,
                                            'n_max': n_max,
                                            'alfa': alfa,
                                            'lamdba': lamdba,
                                        }
                                logging.info('%s: acc: %2f params: %s' % (folder, acc, str(params)))
                                df_run = pd.DataFrame({'Preprocessamento': folder,
                                                        'Acuracia': acc,
                                                        'Classificador': classificador_nome,
                                                        'Hyperparametros': str(params),
                                                        'Execution time': '%.1f' % (time.time() - start_time)
                                                        }, index=[0])
                                df = pd.concat([df, df_run])
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