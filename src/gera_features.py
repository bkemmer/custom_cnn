import pandas as pd
from pathlib import Path
import numpy as np

from extratores import aplica_lbp

def concatenaLBP(path1, path2):
    numPoints=24
    radius=8
    _, hist1 = aplica_lbp(path1)
    _, hist2 = aplica_lbp(path2)
    return np.concatenate((hist1, hist2))

def main():
    data_folder = Path('.', 'Data', 'Processados')
    df_train = pd.read_pickle(Path(data_folder, 'df_train.pickle'))
    df_test = pd.read_pickle(Path(data_folder, 'df_test.pickle'))

    X_train_LBP = df_train.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2']), axis=1).values
    Y_train_LBP = df_train['target'].values

    X_test_LBP = df_test.apply(lambda x: concatenaLBP(x['path_pair_id_1'], x['path_pair_id_2']), axis=1).values
    Y_test_LBP = df_test['target'].values

    X_train_LBP = np.stack(X_train_LBP)
    X_test_LBP = np.stack(X_test_LBP)
    
    output_path = Path(data_folder, 'LBP')
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(Path(output_path, 'X_train_LBP'), X_train_LBP)
    np.save(Path(output_path, 'Y_train_LBP'), Y_train_LBP)
    np.save(Path(output_path, 'X_test_LBP'), X_test_LBP)
    np.save(Path(output_path, 'Y_test_LBP'), Y_test_LBP)

if __name__ == '__main__':
    main()
