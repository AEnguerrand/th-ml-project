import matplotlib.pyplot as plt
import numpy as np


def visualize_single_object(mjd, flux, passbands):
    """plots KPI with a highlight of anomalies
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(mjd, flux,
               marker='.',
               c=passbands)
    plt.show()

    
def visualize_tensor(tensor_object):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(6):
        ax.scatter(np.arange(tensor_object.shape[0]),tensor_object[:,i],marker='.')
    fig.show()