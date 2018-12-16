import pandas as pd
from pickles import pickling
from utils import thread_pool
from tqdm import tqdm

test_set_metadata = False

"""Monothread load chunks (apply and save)"""
def load_apply_save(number_chunks, load_prefix, save_prefix, function, *args, **kwargs):
    pbar = tqdm(total=number_chunks)
    for chunk_id in range(number_chunks):
        pbar.set_description("Processing chunk: %s" % chunk_id)
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        ret_chunk = function(chunk, *args, **kwargs)
        pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)
        pbar.update(1)
    pbar.close()

"""Monothread load chunk (apply only)"""
def load_apply(number_chunks, load_prefix, function, *args, **kwargs):
    pbar = tqdm(total=number_chunks)
    for chunk_id in range(number_chunks):
        pbar.set_description("Processing chunk: %s" % chunk_id)
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        function(chunk, *args, **kwargs)
        pbar.update(1)
    pbar.close()

"""Multithread load chunks (Functions for worker) (apply and save)"""
def load_apply_save_worker_func(chunk_id, load_prefix, save_prefix, function, *args, **kwargs):
    print("Processing chunk:", chunk_id)
    chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
    ret_chunk = function(chunk, *args, **kwargs)
    pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)
    print("Done for chunk:", chunk_id)

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


def convert_test_set_to_chunk():
    chunks = 5000000  # 91 iterations require
    pbar = tqdm(total=91)
    for it_chunk, df_chunk in enumerate(pd.read_csv('dataset/test_set.csv', chunksize=chunks, iterator=True)):
        pbar.set_description("Processing chunk: %s" % it_chunk)
        if it_chunk >= 1:
            df_chunk = pd.concat([df_chunk_cache, df_chunk])
        object_id_last = df_chunk.tail(1)['object_id'].values[0]
        # remove last object_id if is not the end
        if it_chunk != 90:
            df_chunk_cache = df_chunk[df_chunk['object_id'] == object_id_last]
            df_chunk = df_chunk[df_chunk['object_id'] != object_id_last]
        pickling.pickle_chunk_dataframe(df_chunk, it_chunk)
        pbar.update(1)
    pbar.close()

"""Get chunk metadata on tes"""
def get_chunk_metadata_with_chunk_ts_df(chunk):
    object_id_min = chunk.index.levels[0].min()
    object_id_max = chunk.index.levels[0].max()
    res = test_set_metadata.loc[(test_set_metadata['object_id'] >= object_id_min) &
                                (test_set_metadata['object_id'] <= object_id_max)]
    return res


"""Get object_id in the chunk"""
def get_chunk_object_id_with_chunk_ts_df(chunk):
    res = chunk.index.get_level_values('object_id').values
    return res

