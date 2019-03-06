import numpy as np
import pandas as pd

TRAIN_DATA='./train.csv'
TEST_DATA='./test.csv'

def load_train_data():
    df = pd.read_csv(TRAIN_DATA)
    return df

def load_test_data():
    df = pd.read_csv(TEST_DATA)
    return df
