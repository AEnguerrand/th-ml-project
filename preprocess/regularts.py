import numpy as np
import pandas as pd


def gaussian_kernel(difference, tau=5):
    return np.exp(-difference ** 2 / (2 * tau ** 2))


def get_weights(kernel_centers, times, tau=5):
    return gaussian_kernel(np.expand_dims(kernel_centers, 0) - np.expand_dims(times, 1), tau)


def weighted_average(chunk, number_points=None):
    sample_weights = chunk[[f'sw_{i}' for i in range(number_points)]]
    sample_weights /= np.sum(sample_weights, axis=0)
    weighted_flux = np.expand_dims(chunk['flux'].values, 1) * sample_weights.fillna(0)
    return np.sum(weighted_flux, axis=0)


def regularize_dataframe(df, kernel_width=5, kernel_period=20):
    t_min, t_max = df.mjd.min(), df.mjd.max()
    kernel_centers = np.array(np.arange(t_min, t_max, kernel_period))
    weights = get_weights(kernel_centers, df.mjd.values, kernel_width)
    for i in range(len(kernel_centers)):
        df[f'sw_{i}'] = weights[:, i]
    return df.groupby(['object_id', 'passband']).apply(weighted_average, number_points=len(kernel_centers))
