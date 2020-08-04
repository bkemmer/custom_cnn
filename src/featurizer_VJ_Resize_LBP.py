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
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

from utils import read_datasets_X

def aplica_lbp(img, n_points, radius, method, normalize=False, return_lbp=False):
    lbp = local_binary_pattern(img, n_points, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
    if normalize:
        eps = 1e-6
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
    if return_lbp:
        return lbp, hist
    return hist

def aplica_lbph(img, pairs, n_points, radius, method, normalize=False, return_lbp=False):
    hists = []
    for pair in pairs:
        section = img[pair[0][0]:pair[0][1], pair[1][0]:pair[1][1]]
        lbp_section = local_binary_pattern(section, n_points, radius, method=method)
        (hist, _) = np.histogram(lbp_section.ravel(), bins=np.arange(0, n_points + 3))

        if normalize:
            eps = 1e-6
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
        hists.append(hist)

    ret = np.hstack(hists)
    if return_lbp:
        return lbp, ret
    return ret

def geraPairs(img, grid_X, grid_Y):
    w,h = img.shape
    pairs_x = [(grid_X*i, grid_X*(i+1)) for i in range(int(w/grid_X))]
    pairs_y = [(grid_Y*i, grid_Y*(i+1)) for i in range(int(h/grid_Y))]
    pairs = []
    for pair_y in pairs_y:
        for pair_x in pairs_x:
            pairs.append((pair_x, pair_y))
    return pairs

def geraLBPMethod(X_train_1, X_train_2, X_test_1, X_test_2, n_points, radius, lbph=False, 
                    grid_X=None, grid_Y=None, method='uniform', normalize=True):
    # LBPH - divide a imagem em grids e aplica o método LBP somente no grid, e concatena no final
    if lbph:
        if grid_X is None or grid_Y is None:
            logging.error('Grid X ou Y não definida')
            sys.exit()

        X_train_1 = np.array([aplica_lbph(img_1, geraPairs(img_1, grid_X, grid_Y), 
                                            n_points, radius, method) for img_1 in X_train_1])
        X_train_2 = np.array([aplica_lbph(img_2, geraPairs(img_2, grid_X, grid_Y), 
                                            n_points, radius, method) for img_2 in X_train_2])
        X_test_1 = np.array([aplica_lbph(img_1, geraPairs(img_1, grid_X, grid_Y), 
                                            n_points, radius, method) for img_1 in X_test_1])
        X_test_2 = np.array([aplica_lbph(img_2, geraPairs(img_2, grid_X, grid_Y), 
                                            n_points, radius, method) for img_2 in X_test_2])
    # LBP - imagem completa
    else:
        X_train_1 = np.array([aplica_lbp(img_1, n_points, radius, method) for img_1 in X_train_1])
        X_train_2 = np.array([aplica_lbp(img_2, n_points, radius, method) for img_2 in X_train_2])
        X_test_1 = np.array([aplica_lbp(img_1, n_points, radius, method) for img_1 in X_test_1])
        X_test_2 = np.array([aplica_lbp(img_2, n_points, radius, method) for img_2 in X_test_2])

    X_train = np.concatenate((X_train_1, X_train_2), axis=1)
    X_test = np.concatenate((X_test_1, X_test_2), axis=1)

    if normalize:
        X_train = X_train.astype('float64') - np.mean(X_train, axis=0)
        X_test = X_test.astype('float64') - np.mean(X_train, axis=0)
        X_train /= np.std(X_train, axis=0)
        X_test /= np.std(X_train, axis=0)
    
    return X_train, X_test


def main(args):
    if args.size is not None:
        try:
            size = args.size.split('x')
            w = size[0]
            h = size[1]
        except:
            logging.error('Não foi possível ler o tamanho no formato: %s' % args.size)
            sys.exit()

    normalize = args.normalize
    lbph = args.lbph
    if lbph:
        metodologia = 'VJ_resize_LBPH'
        if normalize:
            metodologia = 'VJ_resize_LBPH_normalized'
    else:
        metodologia = 'VJ_resize_LBP'
        if normalize:
            metodologia = 'VJ_resize_LBP_normalized'
    

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.info('Gerando features da metodologia: %s' % metodologia)

    output_folder = Path('.', 'Data', 'features', metodologia)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    data_folder = Path('.', 'Data', 'Processados')
    data_folder = [f for f in data_folder.iterdir() if f.is_dir()]

    for folder in data_folder:
        if folder.name == 'Resultados':
            # Pasta resultados, onde os mesmos são salvos
            continue
        size = folder.name
        size_folder = Path(output_folder, size)
        logging.info('Reading X_train_1, X_train_2, X_test_1, X_test_2 from folder: %s' % folder)
        X_train_1, X_train_2, X_test_1, X_test_2 = read_datasets_X(folder)
        # shape (numero_exemplos, largura, altura)
        method = 'uniform'
        for radius in [1,2,3]:
            for numPoints in [8, 12, 24]:
                if lbph:
                    for grid_X,grid_Y in [(8,8), (16,16)]:
                        logging.info('%s: %s radius: %d numPoint: %d grid:%dx%d' % (metodologia, method, radius, numPoints, grid_X, grid_Y))
                        X_train_LBP, X_test_LBP = geraLBPMethod(X_train_1, X_train_2, X_test_1, X_test_2, 
                                                                numPoints, radius, lbph, grid_X, grid_Y, method, normalize=normalize)
                        experiment_folder = Path(size_folder, '_'.join(['r', str(radius), 'n', str(numPoints), 
                                                                        'grid', str(grid_X), str(grid_Y)]))
                else:
                    logging.info('%s: %s radius: %d numPoint: %d' % (metodologia, method, radius, numPoints))
                    X_train_LBP, X_test_LBP = geraLBPMethod(X_train_1, X_train_2, X_test_1, X_test_2, 
                                                            numPoints, radius, method=method, normalize=normalize)
                    experiment_folder = Path(size_folder, '_'.join(['r', str(radius), 'n', str(numPoints)]))
                experiment_folder.mkdir(parents=True, exist_ok=True)

                np.save(Path(experiment_folder, 'X_train.npy'), X_train_LBP)
                np.save(Path(experiment_folder, 'X_test.npy'), X_test_LBP)

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("-s", "--size", required=False,  
                        help="Tamanho das imagens a serem lidas == nome da pasta onde estão, no formato ex.: 64x128")
    
    ap.add_argument("-lbph", "--lbph", action="store_true", default=False, 
                        help="Flag para gerar o pré-processamento LBPH")
    
    ap.add_argument("-n", "--normalize", action="store_true", default=False, 
                        help="Flag para normalizar usando z-score")

    args = ap.parse_args()
    
    main(args)
