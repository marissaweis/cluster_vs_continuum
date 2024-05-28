import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from utils import sample_from_gaussian_mixture

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int)

path = 'data/ari_neuronal/'
np.random.seed(113)


def main(args):
    n_samples = args.n_samples
    variances = np.array([0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0])
    n_clusters = [7, 10, 15, 20, 40, 60, 80]

    for n in tqdm(n_clusters):
        cluster_means = np.load(Path(path, f'best_means_nc{n}.npy'))
        cluster_weights = np.load(Path(path, f'best_weights_nc{n}.npy'))
        for var in variances:

            gm_params = {'weights': cluster_weights, 'means': cluster_means}
            _ = sample_from_gaussian_mixture(
                gm_params=gm_params, n_components=n, var=var, n=n_samples
            )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
