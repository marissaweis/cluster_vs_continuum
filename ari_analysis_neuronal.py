import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)

path = 'data/'
np.random.seed(236)


def main(args):
    input_file = args.input_file

    # Load data.
    df = pd.read_pickle(input_file)
    print(f'Number of neurons: {len(df)}')

    run_analysis(df)


def run_analysis(df):
    cluster_numbers = [7, 10, 15, 20, 40, 60, 80]

    out_path = Path(path, 'ari_neuronal')
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract latents.
    latents = np.stack(df.latent_emb.values)
    train_latents, test_latents = split_data(latents)

    df_results = pd.DataFrame(columns=['n_clusters', 'ari', 'loglikelihood'])
    for n_clusters in tqdm(cluster_numbers):
        (
            ari,
            scores,
            best_score,
            best_preds,
            best_cluster_means,
            best_cluster_weights,
        ) = run_ari_analysis(
            latents, train_latents, test_latents, n_clusters=n_clusters
        )
        print(f'GMM with {n_clusters} clusters results in ARI: {ari.mean():0.4f}.')
        entry = pd.DataFrame.from_dict(
            {
                'n_clusters': [n_clusters],
                'ari': [ari],
                'loglikelihood': [scores],
            }
        )
        df_results = pd.concat([df_results, entry], ignore_index=True)

        np.save(Path(out_path, f'best_preds_nc{n_clusters}.npy'), best_preds)
        np.save(Path(out_path, f'best_means_nc{n_clusters}.npy'), best_cluster_means)
        np.save(
            Path(out_path, f'best_weights_nc{n_clusters}.npy'), best_cluster_weights
        )
    df_results.to_pickle(Path(out_path, f'results_table_neuron.pkl'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
