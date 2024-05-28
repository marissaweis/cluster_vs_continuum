import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

np.random.seed(236)


def sample_from_gaussian_mixture(
    gm_params: dict,
    n_components: int,
    var: float,
    n: int,
    savepath: str = 'data/synthetic_data',
):
    """Sample from Gaussian mixture model.
    Args:
        gm_params: Dict of str: np.array: Must contain 'weights': (n_components)
            and 'means': (n_components x n_dimensions).
        n_components: Number of components of Gaussian mixture model.
        var: Variance of isotropic Gaussians.
        n: Number of data points to sample.
        savepath: Directory in which to save samples and labels.

    Returns:
        samples: np.array (n x n_dimens
        ions)
    """

    gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
    gmm.weights_ = gm_params['weights']
    gmm.means_ = gm_params['means']
    gmm.covariances_ = np.ones(n_components) * var

    samples, labels = gmm.sample(n)

    if savepath:
        np.save(Path(savepath, f'gm_c{n_components}_var{var}_samples.npy'), samples)
        np.save(Path(savepath, f'gm_c{n_components}_var{var}_labels.npy'), labels)

    return samples


def fit_gmm(
    i: int, train_latents: np.ndarray, test_latents: np.ndarray, n_clusters: int
):
    '''Fit GMM n times to data split.

    Args:
        i: Random state
        train_latents: Array of training samples (n_train_samples x n_dimensions)
        test_latents: Array of test samples (n_test_samples x n_dimensions)
        n_clusters: Number of clusters

    Returns:
        predictions: Array of cluster ids for test samples (n_test_samples x n_dimensions)
        score: Average log-likelihood for test samples
        gmm: Fitted sklearn.mixture.GaussianMixture model
    '''
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        init_params='k-means++',
        random_state=i,
    ).fit(train_latents)
    predictions = gmm.predict(test_latents)
    score = gmm.score(test_latents)

    return predictions, score, gmm


def fit_gmm_parallel(
    latents: np.ndarray,
    train_latents: np.ndarray,
    test_latents: np.ndarray,
    n_clusters: int = None,
    n_runs: int = None,
):
    '''Fit GMM n times to data split in parallel.'''
    NUM_OF_WORKERS = 4
    with Pool(NUM_OF_WORKERS) as pool:
        results = [
            pool.apply_async(fit_gmm, [i, train_latents, test_latents, n_clusters])
            for i in range(n_runs)
        ]
        predictions = np.zeros((n_runs, len(test_latents)))
        cluster_means = np.zeros((n_runs, n_clusters, test_latents.shape[-1]))
        cluster_weights = np.zeros((n_runs, n_clusters))
        scores = np.zeros((n_runs))
        models = []
        for i, result in enumerate(results):
            p, s, gmm = result.get()
            predictions[i] = p
            scores[i] = s
            cluster_means[i] = gmm.means_
            cluster_weights[i] = gmm.weights_
            models.append(gmm)

    # Select best model based on log-likelihood.
    best_i = np.argmax(scores)
    best_score = scores[best_i]
    best_cluster_means = cluster_means[best_i]
    best_cluster_weights = cluster_weights[best_i]

    # Run best model on full dataset.
    best_preds = models[best_i].predict(latents)

    print(f'Best run {best_i} with score = {best_score:0.4f}.')

    return (
        predictions,
        scores,
        best_score,
        best_preds,
        best_cluster_means,
        best_cluster_weights,
    )


def compute_ari(predictions: np.ndarray):
    '''Compute pair-wise ARI between clustering runs and
    return average value.
    '''
    n_eval = len(predictions)
    aris = np.zeros((n_eval, n_eval))
    for i in range(n_eval):
        for j in range(i + 1, n_eval):
            aris[i, j] = adjusted_rand_score(predictions[i], predictions[j])
    ari = aris[np.triu_indices(n_eval, k=1)]
    return ari


def split_data(latents: np.ndarray):
    '''Split data in train and test set.'''
    n_train = int(len(latents) * 0.9)
    n_test = len(latents) - n_train
    idcs = np.arange(len(latents))
    np.random.shuffle(idcs)
    train_latents = latents[idcs[:n_train]]
    test_latents = latents[idcs[n_train:]]

    return train_latents, test_latents


def run_ari_analysis(
    latents: np.ndarray,
    train_latents: np.ndarray,
    test_latents: np.ndarray,
    n_runs: int = 100,
    n_clusters: int = None,
):
    (
        predictions,
        scores,
        best_score,
        best_preds,
        best_cluster_means,
        best_cluster_weights,
    ) = fit_gmm_parallel(
        latents, train_latents, test_latents, n_clusters=n_clusters, n_runs=n_runs
    )
    ari = compute_ari(predictions)
    return ari, scores, best_score, best_preds, best_cluster_means, best_cluster_weights
