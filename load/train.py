import pandas as pd
training_set = None
training_set_metadata = None


def load_set():
    global training_set
    training_set = pd.read_csv('dataset/training_set.csv')
    return training_set


def load_metadata():
    global training_set_metadata
    print("[LOAD] Load metadata for training dataset")
    training_set_metadata = pd.read_csv('dataset/training_set_metadata.csv')
    return training_set_metadata


if __name__ == '__main__':
    load_set()
    load_metadata()
