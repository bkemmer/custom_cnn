import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern

def aplica_lbp(path, numPoints, radius, method):
    img = plt.imread(path)
    lbp = local_binary_pattern(img, numPoints, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    eps = 1e-6
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return lbp, hist

def concatenaLBP(path1, path2, numPoints, radius, method):
    _, hist1 = aplica_lbp(path1, numPoints, radius, method)
    _, hist2 = aplica_lbp(path2, numPoints, radius, method)
    return np.concatenate((hist1, hist2))

def geraLBPMethod(df_train, df_test, output_path, numPoints=24, radius=8, method='uniform'):
    X_train_LBP = df_train.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2'],
                                                        numPoints, radius, method), axis=1).values
    Y_train_LBP = df_train['target'].values

    X_test_LBP = df_test.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2'],
                                                        numPoints, radius, method), axis=1).values
    Y_test_LBP = df_test['target'].values

    X_train_LBP = np.stack(X_train_LBP)
    X_test_LBP = np.stack(X_test_LBP)
    
    np.save(Path(output_path, 'X_train_LBP' + '_' + str(method)), X_train_LBP)
    np.save(Path(output_path, 'Y_train_LBP'  + '_' + str(method)), Y_train_LBP)
    np.save(Path(output_path, 'X_test_LBP' + '_' + str(method)), X_test_LBP)
    np.save(Path(output_path, 'Y_test_LBP' + '_' + str(method)), Y_test_LBP)

def main():
    data_folder = Path('.', 'Data', 'Processados')
    df_train = pd.read_pickle(Path(data_folder, 'df_train.pickle'))
    df_test = pd.read_pickle(Path(data_folder, 'df_test.pickle'))

    output_path = Path(data_folder, 'LBP')
    output_path.mkdir(parents=True, exist_ok=True)
    
    numPoints=24
    radius=8
    for method in ['default', 'ror', 'uniform', 'nri_uniform', 'var']:
        print(method)
        geraLBPMethod(df_train, df_test, output_path, numPoints, radius, method)

if __name__ == '__main__':
    main()
