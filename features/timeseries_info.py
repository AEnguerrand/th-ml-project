from utils import config

def features_ts_for_object_id_set(object_id_set):
    res = object_id_set.groupby('passband')['flux'].apply(lambda x: (x - x.mean()) / x.std())
    return res