import numpy as np
import pandas as pd

MAX_TS_LENGTH = 1095
OBJECT_ID = "object_id"
PASSBAND = "passband"
MJD = "mjd"
FLUX = "flux"


def gaussian_kernel(difference, tau=5):
    return np.exp(-difference ** 2 / (2 * tau ** 2))


def get_weights(kernel_centers, times, tau=5):
    return gaussian_kernel(np.expand_dims(kernel_centers, 0) - np.expand_dims(times, 1), tau)


def weighted_average_on_chunk(chunk, number_points=None):
    sample_weights = chunk[['sw_{}'.format(i) for i in range(number_points)]]
    sample_weights /= np.sum(sample_weights, axis=0)
    weighted_flux = np.expand_dims(chunk['flux'].values, 1) * sample_weights.fillna(0)
    return np.sum(weighted_flux, axis=0)


def regularize_dataframe_custom(df, kernel_width=5, kernel_period=20):
    t_min, t_max = df.mjd.min(), df.mjd.max()
    kernel_centers = np.array(np.arange(t_min, t_max, kernel_period))
    weights = get_weights(kernel_centers, df.mjd.values, kernel_width)
    for i in range(len(kernel_centers)):
        df['sw_{}'.format(i)] = weights[:, i]
    return df.groupby([OBJECT_ID, PASSBAND]).apply(weighted_average_on_chunk, number_points=len(kernel_centers))


def reset_time_offset(dataframe):
    dataframe[MJD] = dataframe.groupby([OBJECT_ID])[MJD].transform(lambda x: (x - x.min()))


def gen_sample_times(sample_interval=20):
    return np.arange(0, MAX_TS_LENGTH + sample_interval, sample_interval)


def weighted_average(df, kernel_centers):
    weights = get_weights(kernel_centers, df[MJD].values)
    ret = pd.Series(np.dot(df[FLUX].values, weights) / np.dot(np.ones(df[FLUX].values.shape), weights), name="n_flux")
    ret = ret.fillna(0)
    ret.index.name = "T"
    return ret


def regularize_dataframe(df, sample_interval=20, kernel_width=5):
    reset_time_offset(df)
    kernel_centers = gen_sample_times(sample_interval=sample_interval)
    return df.groupby([OBJECT_ID, PASSBAND])[MJD, FLUX].apply(weighted_average, kernel_centers)


def tensorize_regular_ts(regular_ts):
    number_objects = regular_ts.index.levels[0].size
    number_passbands = 6
    mat = regular_ts.values.reshape([number_objects, number_passbands, -1])
    return mat.swapaxes(1, 2)
