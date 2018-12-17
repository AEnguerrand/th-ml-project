import pandas as pd
from utils import config


def load_set():
    config.training_set = pd.read_csv('dataset/training_set.csv')


def load_metadata():
    print("[LOAD] Load metadata for training dataset")
    config.training_set_metadata = pd.read_csv('dataset/training_set_metadata.csv')


if __name__ == '__main__':
    load_set()
    load_metadata()
