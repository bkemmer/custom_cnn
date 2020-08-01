
# coding: utf-8

# In[ ]:


import os
from glob import glob
from skimage import feature
import numpy as np
from pathlib import Path
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


'''
Metodologia:
    1- Viola Jones
    2- Resize = 64x128
    3- HOG:
        3780 HOG features
    4- Concatenate
    5- PCA
'''
metodologia = 'VJ_resize_HOG_Concat_PCA'


# In[ ]:


root_path_to_save = Path('..', 'Data', 'features', metodologia)
root_path_to_save.mkdir(parents=True, exist_ok=True)




# ### Steps 1 and 2 already done


root_dirs = glob(os.path.join('..', 'Data', 'Processados', '*'))
for dir_ in root_dirs:
    if os.path.isdir(dir_):
        print('dir: ', dir_)
        X_train_1 = np.load(Path(dir_, 'X_train_1.npy'))
        X_train_2 = np.load(Path(dir_, 'X_train_2.npy'))
        X_test_1 = np.load(Path(dir_, 'X_test_1.npy'))
        X_test_2 = np.load(Path(dir_, 'X_test_2.npy'))
        
        
        # Step 3: apply HOG in pre processed images (VJ + resize)
        X_train_1_HOG = np.array([feature.hog(img_1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                                  block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True, 
                                  multichannel=False) for img_1 in X_train_1])
        
        X_train_2_HOG = np.array([feature.hog(img_2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                                  block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True, 
                                  multichannel=False) for img_2 in X_train_2])
        
        X_test_1_HOG = np.array([feature.hog(img_1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                                  block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True, 
                                  multichannel=False) for img_1 in X_test_1])
        
        X_test_2_HOG = np.array([feature.hog(img_2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                                  block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True, 
                                  multichannel=False) for img_2 in X_test_2])
        
        
        # Step 4: Os vetores X_train1_hog e X_train_2 serao concatenados via coluna para aplicar o PCA 
        X_train_HOG = np.append(X_train_1_HOG, X_train_2_HOG, axis=1)
        X_test_HOG = np.append(X_test_1_HOG, X_test_2_HOG, axis=1)
        
        # Step 5: PCA: components > 0.8
        pca = PCA().fit(X_train_HOG)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.8) + 1
        pca.set_params(**{'n_components':n_components})  
        X_train_PCA = pca.fit_transform(X=X_train_HOG)
        print(X_train_PCA.shape)
        X_test_PCA = pca.transform(X=X_test_HOG)
        print(X_test_PCA.shape)
        
        
        # save files
        image_size = os.path.split(dir_)[-1]
        path_to_save = Path(root_path_to_save, image_size)
        print('Path to save: ', path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)
        np.save(Path(path_to_save, 'X_train.npy'), X_train_PCA)
        np.save(Path(path_to_save, 'X_test.npy'), X_test_PCA)

