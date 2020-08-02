import os
import cv2
import numpy as np
from pathlib import Path

def crop_image(original_image, column, row, width, height):
    # the goal is crop the biggest area
    return original_image[row:row+height, column:column + width]

def open_img(path, color=0):
    '''
    Parameters: 
    - Path: The image should be in the working directory or a full path of image
    should be given;
    - color: Second argument is a flag which specifies the way image should be read.
        cv2.IMREAD_COLOR : Loads a color image. Any transparency of image
        will be neglected;
        cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode;
        cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel;
    Note Instead of these three flags, you can simply pass integers 1, 0 or -1
    respectively.
    Output:
    - img_array: (ndarray)
    '''
    return cv2.imread(path, color)


def save_img(path_img, img):
    '''
    Parameters:
    - path_img: A string representing the file name. The filename must include image format like .jpg, .png, etc.

    - img: It is the image that is to be saved (ndarray).

    Return Value: It returns true if image is saved successfully.
    '''
    cv2.imwrite(path_img, img) 
    
def crop_biggest_area(original_image, detected_faces):
    
    # the goal is crop the biggest area
    if len(detected_faces) == 0: # viola jones didnt recognize any face
        return original_image, (None, None, original_image.shape[0], original_image.shape[1])
    else:
        # detected_faces returns: column, row, width, height
        # So, assuming all width == height
        # get np.argmax of height
        id_max_max_width = np.argmax(detected_faces[:, -1])
        column, row, width, height = detected_faces[id_max_max_width]
        return crop_image(original_image, column, row, width, height), (column, row, width, height)

def detectaFaces(image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), cascade_path=None):
    # utiliza algoritmo proposto por Viola and Jones.
    # return: cropped_image, (column, row, width, height)
    if cascade_path is None:
        cascade_path = os.path.join('..', 'haarcascades', 'haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    rects = face_cascade.detectMultiScale(image,
                scaleFactor = scaleFactor, minNeighbors = minNeighbors,
                minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

    return crop_biggest_area(image, rects)

# def runPreprocessing():
#     df_train['VJ_pair_id_1'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], 
#                                                             path_to_save=x['path_pair_id_1_cropped']), axis=1)
#     df_train['VJ_pair_id_2'] = df_train.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], 
#                                                             path_to_save=x['path_pair_id_2_cropped']), axis=1)

#     df_test['VJ_pair_id_1'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_1'], 
#                                                             path_to_save=x['path_pair_id_1_cropped']), axis=1)
#     df_test['VJ_pair_id_2'] = df_test.apply(lambda x: preprocessing(path_image=x['path_pair_id_2'], 
#                                                             path_to_save=x['path_pair_id_2_cropped']), axis=1)
#     return df_train, df_test

def read_datasets_X(path):
    X_train_1 = np.load(Path(path, 'X_train_1.npy'))
    X_train_2 = np.load(Path(path, 'X_train_2.npy'))
    X_test_1 = np.load(Path(path, 'X_test_1.npy'))
    X_test_2 = np.load(Path(path, 'X_test_2.npy'))
    
    return X_train_1, X_train_2, X_test_1, X_test_2


def padronizar_zscore(X_train, X_val=None, X_test=None):
    if (type(X_train) == 'pandas.core.frame.DataFrame'):

        for col in X_train.columns:
            mean = X_train[col].mean()  # media da coluna nos dados de treino
            std = X_train[col].std(ddof=0)  # std da coluna nos dados de treino
            X_train[col] = X_train[col].apply(lambda x: (x - mean) / std)

            if not X_val is None:
                X_val[col] = X_val[col].apply(
                    lambda x: (x - mean) / std)  # transforma X_val considerando media e desvio do treino
            if not X_test is None:
                X_test[col] = X_test[col].apply(
                    lambda x: (x - mean) / std)  # transforma X_test considerando media e desvio do treino

        return X_train, X_val, X_test, mean, std

    else:
        for index, column in enumerate(X_train.T):
            mean = column.mean()
            std = column.std(ddof=0)
            zscore = lambda t: (t - mean) / std
            X_train[:, index] = np.array([zscore(xi) for xi in column])
            if not X_val is None:
                column_X_val = X_val[:, index]
                X_val[:, index] = np.array([zscore(xi) for xi in column_X_val])
            if not X_test is None:
                column_X_test = X_test[:, index]
                X_test[:, index] = np.array(
                    [zscore(xi) for xi in column_X_test])  # transforma X_test considerando media e desvio do treino

        return X_train, X_val, X_test, mean, std