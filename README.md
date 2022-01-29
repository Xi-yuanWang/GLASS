# GLASS: GNN with Labeling Tricks for Subgraph Representation Learning

This repository is the official implementation of the model in the [following paper](https://openreview.net/forum?id=XLxhEjKNbXj):

Xiyuan Wang, Muhan Zhang. GLASS: GNN with Labeling Tricks for Subgraph Representation Learning. ICLR 2022.

```{bibtex}
@inproceedings{
glass,
title={GLASS: GNN with Labeling Tricks for Subgraph Representation Learning},
author={Xiyuan Wang and Muhan Zhang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=XLxhEjKNbXj}
}
```

#### Install the Environment
You can use conda to create an environment for running the code. 
```{bash}
conda env create --file SubGNN.yml 
```
Before create the environment you may modify the prefix to the location you prefer.

#### Prepare Data

You can download the realworld datasets [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0). Please download, unzip, and put them in ./dataset/. We follow the code provide by [SubGNN](https://github.com/mims-harvard/SubGNN) to produce synthetic datasets. And we also provide the synthetic dataset we use in synds.tar.gz. Please unzip it. We also pretrains SSL embeddings. They are in ./Emb.

The location of each dataset should be
```
CODE
├── Emb
├── dataset
│   ├── em_user
│   ├── hpo_metab
│   ├── ppi_bp
│   └── hpo_neuro
└── dataset_
    ├── density
    ├── coreness
    ├── component
    └── cut_ratio
```
#### Reproduce GLASS
```
mkdir out
python GLASStasks.py
```
You can change the device and dataset in GLASStasks.py.
