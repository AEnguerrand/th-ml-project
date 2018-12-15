import pandas as pd
from pickles import pickling
from utils import thread_pool

"""Monothread load chunks (apply and save)"""
def load_apply_save(number_chunks, load_prefix, save_prefix, function, *args, **kwargs):
    for chunk_id in range(number_chunks):
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        ret_chunk = function(chunk, *args, **kwargs)
        pickling.pickle_chunk(chunk=ret_chunk, chunk_id=chunk_id, prefix=save_prefix)

"""Monothread load chunk (apply only)"""
def load_apply(number_chunks, load_prefix, function, *args, **kwargs):
    for chunk_id in range(number_chunks):
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        function(chunk, *args, **kwargs)


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
