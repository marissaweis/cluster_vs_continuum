import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path

from utils import *

path = 'data/'
np.random.seed(236)

parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int)


def main(args):
    n_c1 = int(args.n_clusters)
    run_analysis(n_c1)


def load_synthetic_data(var, n_c):
    sample = np.load(Path(path, 'synthetic_data', f'gm_c{n_c}_var{var}_samples.npy'))
    return sample


def run_analysis(n_c1):
    variances = np.array([0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0])
    n_clusters = [7, 10, 15, 20, 40, 60, 80]

    df_results = pd.DataFrame(
        columns=[
            'n_clusters_generate',
            'n_clusters_analyze',
            'var',
            'ari',
            'loglikelihood',
        ]
    )
    for var in tqdm(variances):
        latents = load_synthetic_data(var, n_c1)
        train_latents, test_latents = split_data(latents)

        for n_c2 in tqdm(n_clusters):
            df_temp = run_analysis_per_gm(
                latents, train_latents, test_latents, var, n_c1, n_c2
            )
            df_results = pd.concat([df_results, df_temp])

    df_results.to_pickle(
        Path(path, 'ari_synthetic', f'results_table_synthetic_ng{n_c1}.pkl')
    )


def run_analysis_per_gm(latents, train_latents, test_latents, var, n_c1, n_c2):

    out_path = Path(path, 'ari_synthetic')
    out_path.mkdir(parents=True, exist_ok=True)

    ari, scores, best_score, best_preds, best_cluster_means, _ = run_ari_analysis(
        latents, train_latents, test_latents, n_clusters=n_c2
    )
    df = pd.DataFrame.from_dict(
        {
            'n_clusters_generate': [n_c1],
            'n_clusters_analyze': [n_c2],
            'var': [var],
            'ari': [ari],
            'loglikelihood': [scores],
        }
    )
    print(f'With {n_c1=}, {n_c2=} and {var=} results in ARI: {ari.mean():0.4f}.')

    if n_c1 == n_c2:
        np.save(Path(out_path, f'best_preds_nc{n_c1}_var{var}.npy'), best_preds)
        np.save(Path(out_path, f'best_means_nc{n_c1}_var{var}.npy'), best_cluster_means)

    return df


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
