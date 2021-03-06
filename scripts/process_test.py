import pandas as pd
import numpy as np
from load import test
from preprocess import regularts
from pickles import pickling
from datetime import datetime
import sys

CHUNK_DF_PATH_PREFIX = pickling.CHUNK_DF_FILE_PREFIX
CHUNK_TS_DF_PATH_PREFIX = pickling.CHUNK_TS_DF_FILE_PREFIX
CHUNK_TF_PATH_PREFIX = pickling.CHUNK_TF_FILE_PREFIX


def process_test_monothread():
    print('[Process Test Dataset] [Mono Thread] Start ...')
    start_time = datetime.now()
    test.load_apply_save(number_chunks=91, load_prefix=CHUNK_DF_PATH_PREFIX,
                         save_prefix=CHUNK_TS_DF_PATH_PREFIX, function=regularts.regularize_dataframe)
    test.load_apply_save(number_chunks=91, load_prefix=CHUNK_TS_DF_PATH_PREFIX,
                         save_prefix=CHUNK_TF_PATH_PREFIX, function=regularts.tensorize_regular_ts)
    time_elapsed = datetime.now() - start_time
    print('[Process Test Dataset] Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


def process_test_multithread():
    print('[Process Test Dataset] [Multi Thread] Start ...')
    start_time = datetime.now()
    test.load_apply_save_multithreaded(number_chunks=91, load_prefix=CHUNK_DF_PATH_PREFIX,
                                       save_prefix=CHUNK_TS_DF_PATH_PREFIX, function=regularts.regularize_dataframe)
    test.load_apply_save_multithreaded(number_chunks=91, load_prefix=CHUNK_TS_DF_PATH_PREFIX,
                                       save_prefix=CHUNK_TF_PATH_PREFIX, function=regularts.tensorize_regular_ts)
    time_elapsed = datetime.now() - start_time
    print('[Process Test Dataset] Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    process_test_monothread()
