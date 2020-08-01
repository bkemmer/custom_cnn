#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Lê as faces que estavam nas imagens originais, filtradas pelo método
    Viola-Jones + redimensionada para ter um tamanho esperado ex.: 64x128
    Feito isso aplica um método de pré-processamento (LBP e/ou LBPH). 

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

def aplica_lbp(path, numPoints, radius, method):
    img = plt.imread(path)
    lbp = local_binary_pattern(img, numPoints, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3))#, range=(0, numPoints + 2))
    eps = 1e-6
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return lbp, hist

def concatenaLBP(path1, path2, numPoints, radius, method):
    _, hist1 = aplica_lbp(path1, numPoints, radius, method)
    _, hist2 = aplica_lbp(path2, numPoints, radius, method)
    return np.concatenate((hist1, hist2))

def geraLBPMethod(df_train, df_test, output_path, numPoints, radius, method='uniform'):
    X_train_LBP = df_train.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2'],
                                                        numPoints, radius, method), axis=1).values
    Y_train_LBP = df_train['target'].values

    X_test_LBP = df_test.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2'],
                                                        numPoints, radius, method), axis=1).values
    Y_test_LBP = df_test['target'].values

    X_train_LBP = np.stack(X_train_LBP)
    X_test_LBP = np.stack(X_test_LBP)
    
    np.save(Path(output_path, 'X_train_LBP_' + str(radius) + '_' + str(method)), X_train_LBP)
    np.save(Path(output_path, 'Y_train_LBP_' + str(radius) + '_' + str(method)), Y_train_LBP)
    np.save(Path(output_path, 'X_test_LBP_' + str(radius) + '_' + str(method)), X_test_LBP)
    np.save(Path(output_path, 'Y_test_LBP_' + str(radius) + '_' + str(method)), Y_test_LBP)

def main(args):

    try:
        size = args.size.split('x')
        w = size[0]
        h = size[1]
    except:
        logging.error('Não foi possível ler o tamanho no formato: %s' % args.size)
        sys.exit()

    # if args.grid:
    #     try:
    #         grid = args.grid.split('x')
    #         w = size[0]
    #         s = size[1]
    #     except:
    #         logging.error('Não foi possível ler o tamanho no formato: %s' % args.grid)
    #         sys.exit()
    

    data_folder = Path('.', 'Data', 'Processados')
    df_train = pd.read_pickle(Path(data_folder, 'df_train.pickle'))
    df_test = pd.read_pickle(Path(data_folder, 'df_test.pickle'))

    output_path = Path(data_folder, 'LBP')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # radius=3
    # numPoints= 8 * radius

    for radius in [1,2,3]:
        numPoints= 8 * radius:
            print('%s radius: %d' % (method, radius_))
            geraLBPMethod(df_train, df_test, output_path, numPoints, radius_, method)

if __name__ == '__main__':

    # -s 64x128 -lbph -r 1 -n 8
    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("-s", "--size", required=True,  
                        help="Tamanho das imagens a serem lidas == nome da pasta onde estão, no formato ex.: 64x128")
    
    ap.add_argument("-lbph", "--lbph", action="store_true", default=False, 
                        help="Flag para gerar o pré-processamento LBPH")

    # ap.add_argument("-r", "--raio", default=False, 
    #                     help="Raio a ser utilizado na geração do pré-processamento LBP")
    
    # ap.add_argument("-n", "--numeroPontos", default=False, 
    #                     help="Número de pontos a ser utilizado na geração do pré-processamento LBP")
    
    # ap.add_argument("-grid", "--grid", default=False, 
    #                     help="Tamanho do grid a ser usado no LBPH, no formato ex.: 8x8")
    
    args = ap.parse_args()
    
    main(args)
