import pickle

TRAIN_PROCESSED_DF_FILE = "pickles/train/train_df.p"
TRAIN_PROCESSED_TF_FILE = "pickles/train/train_tf.p"
CHUNK_DF_FILE_PREFIX = "pickles/test/df/df_chunk_"
CHUNK_TS_DF_FILE_PREFIX = "pickles/test/ts/ts_chunk_"
CHUNK_TF_FILE_PREFIX = "pickles/test/tf/tf_chunk_"


def pickle_processed_train(dataframe, filename=TRAIN_PROCESSED_DF_FILE):
    with open(filename, 'wb') as outfile:
        pickle.dump(dataframe, outfile)


def unpickle_processed_train(filename=TRAIN_PROCESSED_DF_FILE):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def pickle_processed_train_tf(tensor, filename=TRAIN_PROCESSED_TF_FILE):
    with open(filename, 'wb') as outfile:
        pickle.dump(tensor, outfile)


def unpickle_processed_train_tf(filename=TRAIN_PROCESSED_TF_FILE):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def generate_chunk_file_name(chunk_id, file_prefix):
    return file_prefix + f'{chunk_id}.p'


def pickle_chunk_dataframe(dataframe, object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_DF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(dataframe, outfile)


def unpickle_chunk_dataframe(object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_DF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def pickle_chunk_processed_ts(ts_dataframe, object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_TS_DF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(ts_dataframe, outfile)


def unpickle_chunk_processed_ts(object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_TS_DF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def pickle_chunk_processed_tf(ts_tensor, object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_TF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(ts_tensor, outfile)


def unpickle_chunk_processed_tf(object_id, filename=None):
    if not filename:
        filename = generate_chunk_file_name(object_id, CHUNK_TF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)
