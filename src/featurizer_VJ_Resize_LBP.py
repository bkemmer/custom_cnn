#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Lê as faces que estavam nas imagens originais, filtradas pelo método
    Viola-Jones + redimensionada para ter um tamanho esperado ex.: 64x128
    Feito isso aplica um método de pré-processamento (LBP e/ou LBPH). 
    
    Metodologia:
        1- Viola Jones
        2- Resize
        3- LBP:
             features
        4- Concatenate

Usage:

    python3 gera_features_LBP.py -s 64x128

Returns:
    Salva os arquivos binários do numpy (.npy) na pasta especificada
"""
#  -lbph -r 1 -n 8
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

from utils import read_datasets_X

def aplica_lbp(img, numPoints, radius, method, normalize=False, return_lbp=False):
    lbp = local_binary_pattern(img, numPoints, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3))#, range=(0, numPoints + 2))
    if normalize:
        eps = 1e-6
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
    if return_lbp:
        return lbp, hist
    return hist

def geraLBPMethod(X_train_1, X_train_2, X_test_1, X_test_2, numPoints, radius, method='uniform', normalize=True):

    X_train_1_LBP = np.array([aplica_lbp(img_1, numPoints, radius, method) for img_1 in X_train_1])
    X_train_2_LBP = np.array([aplica_lbp(img_2, numPoints, radius, method) for img_2 in X_train_2])
    X_test_1_LBP = np.array([aplica_lbp(img_1, numPoints, radius, method) for img_1 in X_test_1])
    X_test_2_LBP = np.array([aplica_lbp(img_2, numPoints, radius, method) for img_2 in X_test_1])

    X_train_LBP = np.concatenate((X_train_1_LBP, X_train_2_LBP), axis=1)
    X_test_LBP = np.concatenate((X_test_1_LBP, X_test_2_LBP), axis=1)

    X_train_LBP = X_train_LBP.astype('float64') - np.mean(X_train_LBP, axis=0)
    X_test_LBP = X_test_LBP.astype('float64') - np.mean(X_train_LBP, axis=0)

    if normalize:
        X_train_LBP /= np.std(X_train_LBP, axis=0)
        X_test_LBP /= np.std(X_train_LBP, axis=0)    
    
    return X_train_LBP, X_test_LBP


def main(args):
    try:
        size = args.size.split('x')
        w = size[0]
        h = size[1]
    except:
        logging.error('Não foi possível ler o tamanho no formato: %s' % args.size)
        sys.exit()

    metodologia = 'VJ_resize_LBP'
    output_folder = Path('.', 'Data', 'features', metodologia)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    data_folder = Path('.', 'Data', 'Processados')
    data_folder = [f for f in data_folder.iterdir() if f.is_dir()]

    for folder in data_folder:
        size = folder.name
        size_folder = Path(output_folder, size)

        X_train_1, X_train_2, X_test_1, X_test_2 = read_datasets_X(folder)
        # shape (numero_exemplos, largura, altura)
        method = 'uniform'
        for radius in [1,2,3]:
            for numPoints in [8, 12, 24]:
                logging.info('LBP: %s radius: %d numPoint: %d' % (method, radius, numPoints))
                X_train_LBP, X_test_LBP = geraLBPMethod(X_train_1, X_train_2, X_test_1, X_test_2, 
                                                         numPoints, radius, method, normalize=True)

                experiment_folder = Path(size_folder, '_'.join(['r', str(radius), 'n', str(numPoints)]))
                experiment_folder.mkdir(parents=True, exist_ok=True)

                np.save(Path(experiment_folder, 'X_train.npy'), X_train_LBP)
                np.save(Path(experiment_folder, 'X_test.npy'), X_test_LBP)

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("-s", "--size", required=True,  
                        help="Tamanho das imagens a serem lidas == nome da pasta onde estão, no formato ex.: 64x128")
    
    ap.add_argument("-lbph", "--lbph", action="store_true", default=False, 
                        help="Flag para gerar o pré-processamento LBPH")

    args = ap.parse_args()
    
    main(args)
