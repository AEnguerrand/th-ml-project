import pandas as pd
import numpy as np
from load import test
from preprocess import regularts
from pickles import pickling
from datetime import datetime

TEST_CSV_PATH = 'dataset/test_set.csv'
CHUNK_DF_PATH_PREFIX = pickling.CHUNK_DF_FILE_PREFIX
CHUNK_TS_DF_PATH_PREFIX = pickling.CHUNK_TS_DF_FILE_PREFIX
CHUNK_TF_PATH_PREFIX = pickling.CHUNK_TF_FILE_PREFIX


def process_test_csv_to_pickles():
    chunks = 5000000  # 91 iterations require
    for it_chunk, df_chunk in enumerate(pd.read_csv(TEST_CSV_PATH, chunksize=chunks, iterator=True)):
        print("===================")
        print('iteration:', it_chunk)
        if it_chunk >= 1:
            df_chunk = pd.concat([df_chunk_cache, df_chunk])
        object_id_last = df_chunk.tail(1)['object_id'].values[0]
        # remove last object_id if is not the end
        if it_chunk != 90:
            df_chunk_cache = df_chunk[df_chunk['object_id'] == object_id_last]
            df_chunk = df_chunk[df_chunk['object_id'] != object_id_last]
        pickling.pickle_chunk_dataframe(df_chunk, it_chunk)


def process_test_multithreaded():
    start_time = datetime.now()
    process_test_csv_to_pickles()
    test.load_apply(number_chunks=91, load_prefix=CHUNK_DF_PATH_PREFIX,
                                       save_prefix=CHUNK_TS_DF_PATH_PREFIX, function=regularts.regularize_dataframe)
    test.load_apply(number_chunks=91, load_prefix=CHUNK_TS_DF_PATH_PREFIX,
                                       save_prefix=CHUNK_TF_PATH_PREFIX, function=regularts.tensorize_regular_ts)
    time_elapsed = datetime.now() - start_time
    print('[Process Test Dataset] Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    process_test_multithreaded()
