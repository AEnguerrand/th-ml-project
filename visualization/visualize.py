import matplotlib.pyplot as plt


def visualize_single_object(mjd, flux, passbands):
    """plots KPI with a highlight of anomalies
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(mjd, flux,
               marker='.',
               c=passbands)
    plt.show()
