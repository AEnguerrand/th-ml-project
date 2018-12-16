import pandas as pd
from pickles import pickling
from utils import thread_pool

test_set_metadata = False

"""Monothread load chunks (apply and save)"""
def load_apply_save(number_chunks, load_prefix, save_prefix, function, *args, **kwargs):
    for chunk_id in range(number_chunks):
        print("processing chunk:", chunk_id)
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        ret_chunk = function(chunk, *args, **kwargs)
        pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)
        print("finish for:", chunk_id)

"""Monothread load chunk (apply only)"""
def load_apply(number_chunks, load_prefix, function, *args, **kwargs):
    for chunk_id in range(number_chunks):
        print("processing chunk:", chunk_id)
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        function(chunk, *args, **kwargs)
        print("finish for:", chunk_id)


"""Multithread load chunks (Functions for worker) (apply and save)"""
def load_apply_save_worker_func(chunk_id, load_prefix, save_prefix, function, *args, **kwargs):
    print("processing chunk:", chunk_id)
    chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
    ret_chunk = function(chunk, *args, **kwargs)
    pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)
    print("finish for:", chunk_id)

"""Multithread load chunks (loop) (apply and save)"""
def load_apply_save_multithreaded(number_chunks, load_prefix, save_prefix, function, *args, **kwargs):
    pool = thread_pool.ThreadPool(62)
    for chunk_id in range(number_chunks):
        pool.add_task(load_apply_save_worker_func, chunk_id, load_prefix, save_prefix, function, *args, **kwargs)
    pool.wait_completion()


def load_metadata():
    global test_set_metadata
    print("[LOAD] Load metadata for test dataset")
    test_set_metadata = pd.read_csv('dataset/test_set_metadata.csv')


"""Get chunk metadata on tes"""
def get_chunk_metadata_with_chunk_ts_df(chunk):
    object_id_min = chunk.index.levels[0].min()
    object_id_max = chunk.index.levels[0].max()
    res = test_set_metadata.loc[(test_set_metadata['object_id'] >= object_id_min) &
                                (test_set_metadata['object_id'] <= object_id_max)]
    return res


