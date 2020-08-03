
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
from sklearn.decomposition import PCA




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
        3- Concatenate pair => 64x256
        4- HOG:
            3780 HOG features => 7560
        5- PCA
        6- Normalize
    '''
    metodologia = 'VJ_resize_Concat_HOG_PCA_Normalize'
    output_folder = Path('..', 'Data', 'features', metodologia)
    output_folder .mkdir(parents=True, exist_ok=True)
    cells_per_blocks = [(1,1), (2, 2)]
    # Steps 1 and 2 already done
    data_folder = [f for f in Path('..', 'Data', 'Processados').iterdir() if f.is_dir()]
    for folder in data_folder :
        logging.info('dir: ', folder)
        X_train_1, X_train_2, X_test_1, X_test_2 = utils.read_datasets_X(folder)

        #3 - Concatenate
        X_train = np.append(X_train_1, X_train_2, axis=1)
        X_test = np.append(X_test_1, X_test_2, axis=1)

        for cells_per_block in cells_per_blocks:
            # Step 4 HOG: 3780 HOG features => 7560
            X_train_HOG = np.array([feature.hog(img_1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=cells_per_block,
                                                  block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True,
                                                  multichannel=False) for img_1 in X_train])
            X_test_HOG = np.array([feature.hog(img_1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=cells_per_block,
                                      block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True,
                                      multichannel=False) for img_1 in X_test])

            # Step 5: PCA
            pca = PCA().fit(X_train_HOG)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.8) + 1
            pca.set_params(**{'n_components': n_components})
            X_train_PCA = pca.fit_transform(X=X_train_HOG)
            X_test_PCA = pca.transform(X=X_test_HOG)

            # Step 5: Normalize
            X_train_PCA = X_train_PCA.astype('float64') - np.mean(X_train_PCA, axis=0)
            X_test_PCA = X_test_PCA.astype('float64') - np.mean(X_train_PCA, axis=0)
            X_train_PCA /= np.std(X_train_PCA, axis=0)
            X_test_PCA /= np.std(X_train_PCA, axis=0)



            # save files
            size = folder.name
            size_folder = Path(output_folder, size)

            experiment_folder = Path(size_folder, str(cells_per_block))
            experiment_folder.mkdir(parents=True, exist_ok=True)
            np.save(Path(experiment_folder, 'X_train.npy'), X_train_PCA)
            np.save(Path(experiment_folder, 'X_test.npy'), X_test_PCA)

if __name__ == '__main__':
    main()
