import pickle

TRAIN_PROCESSED_DF_FILE = "pickles/train/train_df.p"
TRAIN_PROCESSED_TF_FILE = "pickles/train/train_tf.p"
SINGLE_OBJECT_DF_FILE_PREFIX = "pickles/test/df/df_single_object_"
SINGLE_OBJECT_TS_DF_FILE_PREFIX = "pickles/test/ts/ts_single_object_"
SINGLE_OBJECT_TF_FILE_PREFIX = "pickles/test/tf/tf_single_object_"


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


def generate_single_object_file_name(object_id, file_prefix):
    return file_prefix + f'{object_id}.p'


def pickle_single_object_dataframe(dataframe, object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_DF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(dataframe, outfile)


def unpickle_single_object_dataframe(object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_DF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def pickle_single_object_processed_ts(ts_dataframe, object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_TS_DF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(ts_dataframe, outfile)


def unpickle_single_object_processed_ts(object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_TS_DF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def pickle_single_object_processed_tf(ts_tensor, object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_TF_FILE_PREFIX)
    with open(filename, 'wb') as outfile:
        pickle.dump(ts_tensor, outfile)


def unpickle_single_object_processed_tf(object_id, filename=None):
    if not filename:
        filename = generate_single_object_file_name(object_id, SINGLE_OBJECT_TF_FILE_PREFIX)
    with open(filename, 'rb') as infile:
        return pickle.load(infile)
