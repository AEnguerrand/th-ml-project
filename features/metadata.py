from utils import config


def features_metadata_for_object_id(object_id, datasetType='Train'):
    if datasetType == 'Train':
        res = config.training_set_metadata[config.training_set_metadata['object_id'] == object_id]
    elif datasetType == 'Test':
        res = config.test_set_metadata[config.test_set_metadata['object_id'] == object_id]
    else:
        print("(features_metadata_for_object_id) Dataset not defined")
        return None
    return res


def features_metadata_useful_for_object_id(object_id, datasetType='Train'):
    metadata = features_metadata_for_object_id(object_id, datasetType)
    del metadata['hostgal_specz'], metadata['ra'], metadata['decl'], metadata['gal_l'], metadata['gal_b'],\
        metadata['ddf']
    return metadata

def features_metadata_useful_all(datasetType='Train'):
    if datasetType == 'Train':
        metadata = config.training_set_metadata
    elif datasetType == 'Test':
        metadata = config.test_set_metadata
    else:
        print("(features_metadata_for_object_id) Dataset not defined")
        return None
    del metadata['hostgal_specz'], metadata['ra'], metadata['decl'], metadata['gal_l'], metadata['gal_b'],\
        metadata['ddf']
    return metadata
