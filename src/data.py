import os
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    data_folder = Path('..', 'Data')
    train_path = Path(data_folder, 'pairsDevTrain.txt')
    with open(train_path) as f:
        train_list = f.readlines()
    test_path = Path(data_folder, 'pairsDevTest.txt')
    with open(test_path) as f:
        test_list = f.readlines()
    
    test_path = Path(data_folder, 'pairsDevTest.txt')

    lfw2_folder = Path(data_folder, 'lfw2')


if __name__ == '__main__':
    main()
