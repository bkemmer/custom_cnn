import numpy as np
from pathlib import Path

def readNpy(fname):
    with open(fname + '.npy', 'rb') as f:
        return np.load(f)

def main():
    data_path = Path('.', 'Data', 'Processados', 'LBP')
    X_train_LBP = readNpy('X_train_LBP')
    Y_train_LBP = readNpy('Y_train_LBP')
    X_test_LBP = readNpy('X_test_LBP')
    Y_test_LBP = readNpy('Y_test_LBP')
    
        

if __name__ == '__main__':
    main()