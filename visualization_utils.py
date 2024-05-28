import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
import colorcet as cc
import matplotlib.pylab as plt
from cmcrameri import cm
import diptest

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

color_dict = {
    0: (0.5843137254901961, 0.7098039215686275, 0.4666666666666667),
    1: (0.0, 0.33725490196078434, 0.34901960784313724),
    2: (0.00392156862745098, 0.5294117647058824, 0.0),
    3: (0.7372549019607844, 0.7137254901960784, 1.0),
    4: (0.027450980392156862, 0.4549019607843137, 0.8470588235294118),
    5: (0.0, 0.0, 0.8666666666666667),
    6: (0.3411764705882353, 0.23137254901960785, 0.0),
    7: (1.0, 0.49411764705882355, 0.8196078431372549),
    8: (0.6313725490196078, 0.4588235294117647, 0.4117647058823529),
    9: (0.0, 0.9921568627450981, 0.8117647058823529),
    10: (1.0, 0.6470588235294118, 0.1843137254901961),
    11: (0.5490196078431373, 0.23137254901960785, 1.0),
    12: (0.4196078431372549, 0.0, 0.30980392156862746),
    13: (0.0, 0.6745098039215687, 0.7764705882352941),
    14: (0.39215686274509803, 0.32941176470588235, 0.4549019607843137),
    15: (0.7490196078431373, 0.011764705882352941, 0.7215686274509804),
    16: (0.8392156862745098, 0.0, 0.0),
    17: (0.4745098039215686, 0.0, 0.0),
    18: (0.592156862745098, 1.0, 0.0),
    19: (0.9921568627450981, 0.9568627450980393, 0.5647058823529412),
}


def remap_cluster_ids(gt_means: np.ndarray, means: np.ndarray):
    """Remap cluster IDs for plotting such that each predicted cluster of the
    synthetic data has same ID as underlying neuronal cluster.
    """
    distances = euclidean_distances(gt_means, means)
    lsa = linear_sum_assignment(distances)
    lsa_dict = {l: i for i, l in zip(*lsa)}
    lsa_dict_inv = {i2: i1 for i1, i2 in lsa_dict.items()}
    return lsa_dict, lsa_dict_inv


def get_knn_dict(means: np.ndarray, k: int = 3, thresh: float = np.inf):
    """Compute k-nearest neighbors of cluster means."""
    knn = kneighbors_graph(means, k, mode='distance', include_self=False).toarray()
    knn[knn > thresh] = 0

    knn_dict = {}
    for i in range(len(means)):
        neighbors = np.where(knn[i])[0]
        knn_dict[i] = set(neighbors)
    return knn_dict


def compute_projection(
    cluster1: int,
    cluster2: int,
    means: np.ndarray,
    latents: np.ndarray,
    predictions: np.ndarray,
):
    """Compute 1D projection of samples onto line running through cluster means.

    Args:
        cluster1: ID of first cluster.
        cluster2: ID of second cluster.
        means: Cluster means (array of size (n_clusters x n_dimensions))
        latents: array of size (n_samples x n_dimensions)
        predictions: Cluster membership (array of size (n_samples))
    """
    c = means[cluster1] - means[cluster2]
    unit_vector = c / np.linalg.norm(c)

    points1 = latents[predictions == cluster1]
    points2 = latents[predictions == cluster2]

    cluster1_proj = np.dot(points1, unit_vector)
    cluster2_proj = np.dot(points2, unit_vector)

    mean = (np.mean(cluster1_proj) + np.mean(cluster2_proj)) / 2

    cluster1_proj -= mean
    cluster2_proj -= mean

    return cluster1_proj, cluster2_proj


def compute_cdf(points: np.ndarray, bins: int = 50):
    """Compute cumulative distribution function for plotting."""
    count, bins_count = np.histogram(points, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count, cdf


def plot_cdf(
    cluster1: int,
    means: np.ndarray,
    latents: np.ndarray,
    predictions: np.ndarray,
    knn: dict,
    color_dict: dict,
    savepath: str = None,
):
    """Plot cumulative distribution function of samples.
    Args:
        cluster1: ID of cluster to plot.
        means: Cluster means (array of size (n_clusters x n_dimensions))
        latents: array of size (n_samples x n_dimensions)
        predictions: Cluster membership (array of size (n_samples))
        knn: dictionary of nearest neighbors per cluster (dict int: set(int))
        color_dict: dictionary of colors per cluster ID
        savepath: str
    """
    # Retrieve nearest neighbors of cluster.
    neighbors = list(knn[cluster1])

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    cluster1_proj, cluster2_proj = compute_projection(
        cluster1, neighbors[0], means, latents, predictions
    )
    bc, cdf = compute_cdf(np.concatenate([cluster1_proj, cluster2_proj]) * -1, bins=50)

    ax.plot(
        bc[1:],
        cdf,
        label=f'{cluster1}_{neighbors[0]}',
        color='black',
        linewidth=3,
        alpha=0.75,
    )

    ax.tick_params('y', labelbottom=True, width=2, labelsize=12)
    ax.set_xticks([-2, 0, 2])
    ax.set_title('Joint CDF', fontsize=12)
    ax.tick_params('x', labelleft=True, width=2, labelsize=12)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    sns.despine(trim=1)
    plt.savefig(Path(savepath), bbox_inches='tight', transparent=True)
    plt.close()


def plt_projection_hists(
    cluster1: int,
    means: np.ndarray,
    latents: np.ndarray,
    predictions: np.ndarray,
    knn: dict,
    color_dict: dict,
    savepath: str = None,
):
    """Plot projection of two clusters onto the line connecting their means.
    Args:
        cluster1: ID of cluster to plot.
        means: Cluster means (array of size (n_clusters x n_dimensions))
        latents: array of size (n_samples x n_dimensions)
        predictions: Cluster membership (array of size (n_samples))
        knn: dictionary of nearest neighbors per cluster (dict int: set(int))
        color_dict: dictionary of colors per cluster ID
        savepath: str
    """
    neighbors = list(knn[cluster1])

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    cluster1_proj, cluster2_proj = compute_projection(
        cluster1, neighbors[0], means, latents, predictions
    )

    colors = [color_dict[cluster1], color_dict[neighbors[0]]]
    samples = [cluster1_proj * -1, cluster2_proj * -1]
    ax.hist(samples, bins=30, alpha=0.75, histtype='bar', stacked=True, color=colors)

    dip = diptest.dipstat(np.concatenate([cluster1_proj, cluster2_proj]))
    ax.set_title(f'Clusters {cluster1} vs. {neighbors[0]}, \n {dip=:.4f}')

    ax.tick_params('y', labelbottom=True, width=2, labelsize=12)
    ax.set_xticks([-2, 0, 2])
    ax.tick_params('x', labelbottom=True, width=2, labelsize=12)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    sns.despine(trim=1)

    plt.savefig(Path(savepath), bbox_inches='tight', transparent=True)
    plt.close()

    return dip


def get_lines(
    cluster1: int,
    means: np.ndarray,
    latents: np.ndarray,
    predictions: np.ndarray,
    knn: dict,
    neigh: int = 0,
):
    """Calculate start and end as well as width of lines connecting two cluster means based on the dip statistic."""
    neighbors = list(knn[cluster1])

    cluster1_proj, cluster2_proj = compute_projection(
        cluster1, neighbors[neigh], means, latents, predictions
    )
    dip = diptest.dipstat(np.concatenate([cluster1_proj, cluster2_proj]))

    return dip, neighbors[neigh]
