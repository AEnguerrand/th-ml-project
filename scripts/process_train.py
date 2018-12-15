import pandas as pd
import numpy as np
from preprocess import regularts
from pickles import pickling
from datetime import datetime

TRAIN_CSV_PATH = '../dataset/training_set.csv'
TRAIN_TS_PATH = '../' + pickling.TRAIN_PROCESSED_DF_FILE
TRAIN_TF_PATH = '../' + pickling.TRAIN_PROCESSED_TF_FILE


def process_train(filename=TRAIN_CSV_PATH):
    start_time = datetime.now()
    print(f'Loading csv at {filename}')
    training_set = pd.read_csv(filename)
    print('Regularizing train TS')
    ts = regularts.regularize_dataframe(training_set)
    print(f'Pickling regularized time series at {TRAIN_TS_PATH}')
    pickling.pickle_processed_train(ts,filename=TRAIN_TS_PATH)
    print('Tensorizing time series')
    tf = regularts.tensorize_regular_ts(ts)
    print(f'Pickling tensorized time series at {TRAIN_TF_PATH}')
    pickling.pickle_processed_train(tf, filename=TRAIN_TF_PATH)
    print('[Process Train Dataset] Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == "__main__":
    process_train()