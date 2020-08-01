import os
from glob import glob
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from utils import detectaFaces
import matplotlib.pyplot as plt

import cv2

from utils import open_img, save_img

def getDF(path):
    with open(path) as f:
        file_list = f.readlines()
    n = int(file_list[0].strip())
    df_inicial = pd.read_csv(path, sep='\t', skiprows=1, nrows=n, names=['pair_name_1', 'pair_id_1', 'pair_id_2'])
    df_inicial['pair_name_2'] = None
    df_secondary = pd.read_csv(path, sep='\t', skiprows=n+1, names=['pair_name_1', 'pair_id_1', 'pair_name_2', 'pair_id_2'])
    df = pd.concat([df_inicial, df_secondary])
    df = df.reset_index(drop=True)
    print(df.shape)
    return df

def preprocessing(path_image, path_to_save, dim=(100, 100)):
    # print(path_image + " START")
    original_image = open_img(path_image, color=0)
    grayscale_image = original_image.copy()
    cropped_image, (column, row, width, height) = detectaFaces(grayscale_image)
    resized = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)
    save_img(path_img=path_to_save, img=resized)
    # print(path_image + " FINISH")
    return (column, row, width, height)

def image_path(person, id_, data_path):
    return glob(os.path.join(data_path, person, '*' + id_ + '.jpg'))[0]

def completaPaths(df_train, df_test, data_path):
    df_train['path_pair_id_1'] = df_train.apply(lambda x: image_path(person=x['pair_name_1'], 
                                                          id_= str(x['pair_id_1']),
                                                          data_path=data_path), axis=1)
    df_train['path_pair_id_2'] = df_train.apply(lambda x: image_path(person=x['pair_name_1'], 
                                                                    id_= str(x['pair_id_2']),
                                                                    data_path=data_path) 
                                                        if x['pair_name_2']==None 
                                                        else image_path(person=x['pair_name_2'], 
                                                                    id_= str(x['pair_id_2']),
                                                                    data_path=data_path), axis=1)

    df_test['path_pair_id_1'] = df_test.apply(lambda x: image_path(person=x['pair_name_1'], 
                                                                    id_= str(x['pair_id_1']),
                                                                    data_path=data_path), axis=1)
    df_test['path_pair_id_2'] = df_test.apply(lambda x: image_path(person=x['pair_name_1'], 
                                                                    id_= str(x['pair_id_2']),
                                                                    data_path=data_path) 
                                                        if x['pair_name_2']==None 
                                                        else image_path(person=x['pair_name_2'], 
                                                                        id_= str(x['pair_id_2']),
                                                                        data_path=data_path), axis=1)
    return df_train, df_test

def createDirs(df_train, df_test):
    cropped_dir = os.path.join('..', 'Data', 'lfw2_cropped')
    if not os.path.isdir(cropped_dir):
        os.mkdir(cropped_dir)
    df_train['path_pair_id_1_cropped'] = df_train['path_pair_id_1'].apply(lambda x: x.replace('lfw2', 'lfw2_cropped'))
    _ = df_train['path_pair_id_1_cropped'].apply(lambda x: None if os.path.isdir(os.path.split(x)[0]) 
                                                                else os.mkdir(os.path.split(x)[0]))

    df_train['path_pair_id_2_cropped'] = df_train['path_pair_id_2'].apply(lambda x: x.replace('lfw2', 'lfw2_cropped'))
    _ = df_train['path_pair_id_2_cropped'].apply(lambda x: None if os.path.isdir(os.path.split(x)[0]) 
                                                                else os.mkdir(os.path.split(x)[0]))
    
    df_test['path_pair_id_1_cropped'] = df_test['path_pair_id_1'].apply(lambda x: x.replace('lfw2', 'lfw2_cropped'))
    _ = df_test['path_pair_id_1_cropped'].apply(lambda x: None if os.path.isdir(os.path.split(x)[0]) 
                                                                else os.mkdir(os.path.split(x)[0]))

    df_test['path_pair_id_2_cropped'] = df_test['path_pair_id_2'].apply(lambda x: x.replace('lfw2', 'lfw2_cropped'))
    _ = df_test['path_pair_id_2_cropped'].apply(lambda x: None if os.path.isdir(os.path.split(x)[0]) 
                                                                else os.mkdir(os.path.split(x)[0]))
    
    return df_train, df_test

def executaPreprocessamento(df_train, df_test, dim=(100, 100)):
    df_train['VJ_pair_id_1'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], path_to_save=x['path_pair_id_1_cropped'], dim=dim), axis=1)
    df_train['VJ_pair_id_2'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], path_to_save=x['path_pair_id_2_cropped'], dim=dim), axis=1)

    df_test['VJ_pair_id_1'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], path_to_save=x['path_pair_id_1_cropped'], dim=dim), axis=1)
    df_test['VJ_pair_id_2'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], path_to_save=x['path_pair_id_2_cropped'], dim=dim), axis=1)
    
    return df_train, df_test

def geraVariavelBinaria(df_train, df_test):
    df_train['target'] = df_train.apply(lambda x: 1 if x['pair_name_2'] is None else 0, axis=1)
    df_test['target'] = df_test.apply(lambda x: 1 if x['pair_name_2'] is None else 0, axis=1)
    return df_train, df_test


def gera_dataset(df_):
    X_1 = np.array([open_img(img_1) for img_1 in df_.loc[:, 'path_pair_id_1_cropped'].values])
    X_2 = np.array([open_img(img_2) for img_2 in df_.loc[:, 'path_pair_id_2_cropped'].values])
    return X_1, X_2

def main():
    '''
        Este pipeline é responsável por:
        1- Ler o arquivo pairsDevTrain e pairsDevTest para formar os conjuntos de dados experimentais;
        2- Dado o passo 1, construir um dataframe para pegar o path de cada imagem para construir o par de treinamento
        3- Determinar e criar o diretório que as imagens serão salvas
        4- Ler cada imagem original (diretorio original) executar o pre-processamento e salvar no diretório passo 3
            4.1- Pré-processamento consiste de dois passos: VJ e resize
        5- Construir e salvar as labels, além do dataframe
        6- Salvar X_train e X_test no path correto
    '''

    args = sys.argv
    n_rows = int(args[1])
    n_cols = int(args[2])
    # 1
    data_folder = os.path.join('..', 'Data')
    lfw2 = Path(data_folder, 'lfw2')
    train_path = Path(data_folder, 'pairsDevTrain.txt')
    test_path = Path(data_folder, 'pairsDevTest.txt')
    df_train = getDF(train_path)
    df_test = getDF(test_path)
    #2
    df_train, df_test = completaPaths(df_train, df_test, data_path=lfw2)
    #3
    df_train, df_test = createDirs(df_train, df_test)

    #4.1
    df_train, df_test = executaPreprocessamento(df_train, df_test, dim=(n_rows,n_cols))
    #5
    df_train, df_test = geraVariavelBinaria(df_train, df_test)
    y_train = df_train['target'].values
    y_test = df_test['target'].values
    root_output_path = Path(os.path.join(data_folder, 'Processados'))
    root_output_path.mkdir(parents=True, exist_ok=True)
    df_train.to_pickle(Path(root_output_path, 'df_train.pickle'))
    df_test.to_pickle(Path(root_output_path, 'df_test.pickle'))
    np.save(Path(root_output_path, 'y_train.npy'), y_train)
    np.save(Path(root_output_path, 'y_test.npy'), y_test)

    #6
    output_path = Path(os.path.join(root_output_path, str(n_rows)+'_'+str(n_cols)))
    output_path.mkdir(parents=True, exist_ok=True)
    X_train_1, X_train_2 = gera_dataset(df_train)
    np.save(Path(output_path, 'X_train_1.npy'), X_train_1)
    np.save(Path(output_path, 'X_train_2.npy'), X_train_2)
    X_test_1, X_test_2 = gera_dataset(df_test)
    np.save(Path(output_path, 'X_test_1.npy'), X_test_1)
    np.save(Path(output_path, 'X_test_2.npy'), X_test_2)



if __name__ == '__main__':
    main()
