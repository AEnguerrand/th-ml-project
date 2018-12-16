import pandas as pd
training_set = False
training_set_metadata = False


def load_set():
    global training_set
    training_set = pd.read_csv('dataset/training_set.csv')


def load_metadata():
    global training_set_metadata
    print("[LOAD] Load metadata for training dataset")
    training_set_metadata = pd.read_csv('dataset/training_set_metadata.csv')


if __name__ == '__main__':
    load_set()
    load_metadata()
