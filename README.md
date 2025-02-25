# Cluster versus continuum analysis

This repo contains the code for the cluster versus continuum analysis of the paper ["An unsupervised map of excitatory neurons’ dendritic morphology in the mouse visual cortex"](https://doi.org/10.1101/2022.12.22.521541).

![Figure 3](Fig3.png?raw=true "Title")

## ARI analysis of neuronal data
Run the ARI analysis on the neuronal data:
```
python3 ari_analysis_neuronal.py data/neuronal_data/df_neuron.pkl
```

## Generate synthetic data
To generate synthetic data, first run the ARI analysis of the neuronal data to extract the cluster means of the neuronal data. 
Then run with the desired number of samples *N*: 
```
python3 generate_synthetic_data.py --n_samples <N>
```

## ARI analysis of synthetic data
Run the ARI analysis on the synthetic data. Specify the number of ground truth clusters in the synthetic data using the *--n_clusters* argument:
```
python3 ari_analysis_synthetic.py --n_clusters 20
```

## Visualizations
The folder [visualization](https://github.com/marissaweis/cluster_vs_continuum/blob/main/visualization/) contains jupyter notebooks to plot subfigures A - G of Figure 3 of the paper.
Dip statistics in the paper are scaled by a factor of 2 compared to the output of the *diptest* package.


## Citation
```
@article{Weis2024,
      title={An unsupervised map of excitatory neurons' dendritic morphology in the mouse visual cortex},
      author = {Weis, Marissa A. and Papadopoulos, Stelios and Hansel, Laura and Lüddecke, Timo and Celii, Brendan and Fahey, Paul G. and Wang, Eric Y. and MICrONS Consortium and Reimer, Jacob and Berens, Philipp and Tolias, Andreas S. and Ecker, Alexander S.}
      journal={bioRxiv},
      doi = {10.1101/2022.12.22.521541},
      year={2024}
}
```
