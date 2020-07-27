import os
from glob import glob

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

def executaPreprocessamento(df_train, df_test):
    df_train['VJ_pair_id_1'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], path_to_save=x['path_pair_id_1_cropped']), axis=1)
    df_train['VJ_pair_id_2'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], path_to_save=x['path_pair_id_2_cropped']), axis=1)

    df_test['VJ_pair_id_1'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], path_to_save=x['path_pair_id_1_cropped']), axis=1)
    df_test['VJ_pair_id_2'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], path_to_save=x['path_pair_id_2_cropped']), axis=1)
    
    return df_train, df_test

def geraVariavelBinaria(df_train, df_test):
    df_train['target'] = df_train.apply(lambda x: 0 if x['pair_name_2'] is None else 1, axis=1)
    df_test['target'] = df_test.apply(lambda x: 0 if x['pair_name_2'] is None else 1, axis=1)
    return df_train, df_test

def main():

    # data_folder = os.path.abspath('..\\data\\')
    data_folder = os.path.join('.', 'Data')
    lfw2 = Path(data_folder, 'lfw2')
    train_path = Path(data_folder, 'pairsDevTrain.txt')
    test_path = Path(data_folder, 'pairsDevTest.txt')

    df_train = getDF(train_path)
    df_test = getDF(test_path)

    df_train, df_test = completaPaths(df_train, df_test, data_path=lfw2)
    df_train, df_test = createDirs(df_train, df_test)
    df_train, df_test = executaPreprocessamento(df_train, df_test)

    df_train, df_test = geraVariavelBinaria(df_train, df_test)
    
    output_path = Path('Data', 'Processados')
    output_path.mkdir(parents=True, exist_ok=True)
    df_train.to_pickle(Path(output_path, 'df_train.pickle'))
    df_test.to_pickle(Path(output_path, 'df_test.pickle'))



if __name__ == '__main__':
    main()
