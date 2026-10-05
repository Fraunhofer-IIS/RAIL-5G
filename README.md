# RAIL-5G

This is the code for our paper **RAIL-5G: A Challenging Real-World Dataset and Benchmark for AI-Based 5G Indoor Localization**. 
You can download the dataset from [here](https://iis.fraunhofer.de/rail-5g).

## Implemented Methods

This repository implements the following dimensionality reduction and channel charting methods:

- **Fingerprinting** (FP)
- **Principal Component Analysis** (PCA)
- **Isomap** — Tenenbaum, J. B., Silva, V. D., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319-2323.
- **t-SNE** — Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9(11).
- **UMAP** — McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.
- **Autoencoder (AE)**
- **Constrained AE** — Huang, P., Castañeda, O., Gönültaş, E., Medjkouh, S., Tirkkonen, O., Goldstein, T., & Studer, C. (2019, July). Improving channel charting with representation-constrained autoencoders. In *2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)* (pp. 1-5). IEEE.
- **UAGeo** — Euchner, F., Stephan, P., & ten Brink, S. (2024, October). Uncertainty-aware dimensionality reduction for channel charting with geodesic loss. In *2024 58th Asilomar Conference on Signals, Systems, and Computers* (pp. 1665-1672). IEEE.
- **PSSN** — Zhao, L., Yang, Y., Xiong, Q., Wang, H., Yu, B., Sun, F., & Sun, C. (2024, June). A signature based approach towards global channel charting with ultra low complexity. In *2024 IEEE International Conference on Communications Workshops (ICC Workshops)* (pp. 667-672). IEEE.
- **Siamese** — Lei, E., Castañeda, O., Tirkkonen, O., Goldstein, T., & Studer, C. (2019, September). Siamese neural networks for wireless positioning and channel charting. In *2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton)* (pp. 200-207). IEEE.
- **Siamese-G** — Stahlke, M., Yammine, G., Feigl, T., Eskofier, B. M., & Mutschler, C. (2023). Indoor localization with robust global channel charting: A time-distance-based approach. *IEEE Transactions on Machine Learning in Communications and Networking*, 1, 3-17.
- **Triplet** — Ferrand, P., Decurninge, A., Ordonez, L. G., & Guillaud, M. (2021). Triplet-based wireless channel charting: Architecture and experiments. *IEEE Journal on Selected Areas in Communications*, 39(8), 2361-2373.

## Setup the environment

1. Use the python environment manager of your choice to create a new environment and activate it, e.g. ```python -m venv .venv``` and ```source .venv/bin/activate```
2. Install the package and its dependencies using ```pip install -e .```. NOTE: We did not pin any package versions in the setup.py. If you run into compatibility issues in newer versions of some libraries, you can use the versions in the *requirements.txt* file (```pip install -r requirements.txt```).


## Preparing the dataset

1. Download the data to the data subfolder, so that the structure looks like this:
```
data
├── take1.h5
├── take2.h5
├── take3.h5
├── take4.h5
├── take5.h5
├── take6.h5
├── take7.h5
└── take8.h5
```

2. Create dataset splits

```
# Full splits
python run/merge_ds.py --base-folder data --ds-indices 1,2,3,4,5,6 --output-file data/full_train.h5
python run/merge_ds.py --base-folder data --ds-indices 8 --output-file data/full_small_train.h5
python run/merge_ds.py --base-folder data --ds-indices 7 --output-file data/full_test.h5

# Open
python run/merge_ds.py --base-folder data --ds-indices 5 --output-file data/open_train.h5
python run/merge_ds.py --base-folder data --ds-indices 2 --output-file data/open_test.h5

# Lower corridor
python run/merge_ds.py --base-folder data --ds-indices 4 --output-file data/corridor_train.h5
python run/merge_ds.py --base-folder data --ds-indices 1 --output-file data/corridor_test.h5
```

Note that we had an error in our script that would expand the float32 values of some data fields to float64 in the merge_ds.py script. If you run this code for any other purpose than experiment reproduction, we suggest adding the ```--optimized``` flag, which will significantly reduce dataset size, but will also produce slightly different results than reported in the paper

## Create distance metric datasets

```
for cfg in corridor_cira.yaml corridor_cira_g.yaml corridor_signature.yaml full_cira.yaml full_cira_g.yaml full_signature.yaml full_small_cira.yaml open_cira.yaml open_cira_g.yaml open_signature.yaml; do python run/dist_metric.py --config-name "$cfg"; done
```

## Run experiments

You can run all channel charting experiments with the following command.

```
bash run/run_paper.sh
```

## Citation

If you use this dataset in your work, please consider citing 
```
@inproceedings{pirkl2026rail-5g,
    title={RAIL-5G: A Challenging Real-World Dataset and Benchmark for AI-Based 5G Indoor Localization},
    author={Pirkl, Jonas and Eidloth, Andreas and Kasparek, Maximilian and Feigl, Tobias and Mutschler, Christopher},
    booktitle={2026 16th International Conference on Indoor Positioning and Indoor Navigation (IPIN)},
    pages={1--6},
    year={2026},
    organization={IEEE}
}
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
