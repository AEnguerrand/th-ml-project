import pandas as pd
from pickles import pickling
from utils import thread_pool

NUMBER_THREADS = 4


def load_apply_save_worker_func(chunk_id, load_prefix, save_prefix, function, *args, **kwargs):
    print(f"processing chunk {chunk_id}")
    chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
    ret_chunk = function(chunk, *args, **kwargs)
    pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)


def load_apply_save(number_chunks, load_prefix, save_prefix, function, *args, **kwargs):
    pool = thread_pool.ThreadPool(4)
    for chunk_id in range(number_chunks):
        pool.add_task(load_apply_save_worker_func, chunk_id, load_prefix, save_prefix, function, *args, **kwargs)
    pool.wait_completion()


def load_apply(number_chunks, load_prefix, function, *args, **kwargs):
    for chunk_id in range(number_chunks):
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        function(chunk, *args, **kwargs)


def load_pickle():
    chunks = 5000000  # 90 iterations require
    for it_chunk, df_chunk in enumerate(pd.read_csv('./dataset/test_set.csv', chunksize=chunks, iterator=True)):
        print("===================")
        print('iteration:', it_chunk)
        if it_chunk >= 1:
            df_chunk = pd.concat([df_chunk_cache, df_chunk])
        object_id_last = df_chunk.tail(1)['object_id'].values[0]
        # remove last object_id if is not the end
        if it_chunk != 90:
            df_chunk_cache = df_chunk[df_chunk['object_id'] == object_id_last]
            df_chunk = df_chunk[df_chunk['object_id'] != object_id_last]
        # df_chunck contain all data of object_id inside
        # put code here
        pickling.pickle_chunk_dataframe(df_chunk, it_chunk)
