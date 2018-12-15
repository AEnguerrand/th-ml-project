import pandas as pd
import numpy as np
from load import test
from preprocess import regularts
from pickles import pickling
from datetime import datetime

start_time = datetime.now()

test.load_apply_save(number_chunks=91,load_prefix=pickling.CHUNK_DF_FILE_PREFIX,save_prefix=pickling.CHUNK_TS_DF_FILE_PREFIX,function=regularts.regularize_dataframe)
test.load_apply_save(number_chunks=91,load_prefix=pickling.CHUNK_TS_DF_FILE_PREFIX,save_prefix=pickling.CHUNK_TF_FILE_PREFIX,function=regularts.tensorize_regular_ts)

time_elapsed = datetime.now() - start_time
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
