import pandas
from pickles import pickling

def load_apply_save(number_chunks,load_prefix,save_prefix,function,*args,**kwargs):
    for chunk_id in range(number_chunks):
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id, prefix=load_prefix)
        ret_chunk = function(chunk,*args,**kwargs)
        pickling.pickle_chunk(chunk=ret_chunk,chunk_id=chunk_id,prefix=save_prefix)


def load_apply(number_chunks,load_prefix,function,*args,**kwargs):
    for chunk_id in range(number_chunks):
        chunk = pickling.unpickle_chunk(chunk_id=chunk_id,prefix=load_prefix)
        function(chunk,*args,**kwargs)