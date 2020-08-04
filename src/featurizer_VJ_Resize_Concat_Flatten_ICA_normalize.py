
# coding: utf-8

# In[ ]:


import os
from glob import glob
from skimage import feature
import numpy as np
from pathlib import Path
import argparse
import logging,sys
import utils
from sklearn.decomposition import FastICA



# # Paper: Face Recognition Based on HOG and Fast PCA Algorithm
# 
#     1- Viola Jones
#     2- Resize = 64x128
#     3- HOG:
#         3780 HOG features: orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
#     4- PCA
#     5- Normalização: median normalization method (Eq 10)
#     

# In[ ]:

def main():

    '''
    Metodologia:
        1- Viola Jones
        2- Resize = 64x128
        3- Create pair and Concatenate
            3.2- Flatten
        4- ICA -> ica.fit_transform() -> data train
            4.2- process to rollback (for ilustration) -> ica.inverse_transform() and reshape()
        5- Normalize
    '''
    metodologia = 'VJ_resize_Concat_Flatten_ICA_Normalize'
    output_folder = Path('..', 'Data', 'features', metodologia)
    output_folder .mkdir(parents=True, exist_ok=True)
    n_components = [100, 200, 300]
    # Steps 1 and 2 already done
    data_folder = [f for f in Path('..', 'Data', 'Processados').iterdir() if f.is_dir()]
    for folder in data_folder :
        logging.info('dir: ', folder)
        X_train_1, X_train_2, X_test_1, X_test_2 = utils.read_datasets_X(folder)

        #3 - Create pair and Concatenate
        X_train = np.append(X_train_1, X_train_2, axis=1)
        X_test = np.append(X_test_1, X_test_2, axis=1)

        #3.2 - Flatten
        X_train_flatten = np.array([img.flatten() for img in X_train])
        X_test_flatten = np.array([img.flatten() for img in X_test])

        for n_component in n_components:
            # Step 4: ICA
            ica = FastICA(n_components = n_component)
            X_train_flatten_ICA = ica.fit_transform(X_train_flatten)  # train data
            X_test_flatten_ICA = ica.transform(X_test_flatten)  # test data

            # Step 5: Normalize
            X_train_flatten_ICA = X_train_flatten_ICA.astype('float64') - np.mean(X_train_flatten_ICA, axis=0)
            X_test_flatten_ICA = X_test_flatten_ICA.astype('float64') - np.mean(X_train_flatten_ICA, axis=0)
            X_train_flatten_ICA /= np.std(X_train_flatten_ICA, axis=0)
            X_test_flatten_ICA /= np.std(X_train_flatten_ICA, axis=0)


            # save files
            size = folder.name
            size_folder = Path(output_folder, size)

            experiment_folder = Path(size_folder, str(n_component))
            experiment_folder.mkdir(parents=True, exist_ok=True)
            np.save(Path(experiment_folder, 'X_train.npy'), X_train_flatten_ICA)
            np.save(Path(experiment_folder, 'X_test.npy'), X_test_flatten_ICA)

if __name__ == '__main__':
    main()
