import numpy as np
from pathlib import Path
from sklearn import svm



def readNpy(basePath, fname):
    with open(Path(basePath, fname + '.npy'), 'rb') as f:
        return np.load(f)

def main():
    data_path = Path('.', 'Data', 'Processados', 'LBP')

    for radius_ in range(5):
        for method in ['default', 'ror', 'uniform', 'nri_uniform', 'var']:
            X_train_LBP = readNpy(data_path, 'X_train_LBP_' + str(radius_) + '_' + str(method))
            Y_train_LBP = readNpy(data_path, 'Y_train_LBP_' + str(radius_) + '_' + str(method))
            X_test_LBP = readNpy(data_path, 'X_test_LBP_' + str(radius_) + '_' + str(method))
            Y_test_LBP = readNpy(data_path, 'Y_test_LBP_' + str(radius_) + '_' + str(method))
            
            clf = svm.SVC()
            clf.fit(X_train_LBP, Y_train_LBP)
            yHat = clf.predict(X_test_LBP)
            acc = (yHat == Y_test_LBP).sum()/len(yHat)
            print('%s - radius[%d]: %.2f' % (method, radius_, acc))

if __name__ == '__main__':
    main()