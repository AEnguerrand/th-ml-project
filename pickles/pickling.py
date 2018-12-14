import pickle

TRAIN_PROCESSED_FILE = "pickles/train/train.p"

def pickle_processed_train(dataframe,filename = TRAIN_PROCESSED_FILE):
    with open(filename,'wb') as outfile :
        pickle.dump(dataframe,outfile)
        
def unpickle_processed_train(filename = TRAIN_PROCESSED_FILE):
    with open(filename,'rb') as infile :
        return pickle.load(infile)

    