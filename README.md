## GLASS
#### Install the Environment
You can use conda to create an environment for running the code. 
```{bash}
conda env create --file SubGNN.yml 
```
Before create the environment you may modify the prefix to the location you prefer.

#### Prepare Data

You can download the realworld datasets [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0). Please download, unzip, and put them in ./dataset/. We follow the code provide by [SubGNN](https://github.com/mims-harvard/SubGNN). And we also provide the synthetic dataset we use in synds.tar.gz. Please unzip it.

The location of each dataset should be
CODE
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

#### Reproduce GLASS
```
mkdir out
python GLASStasks.py
```
You can change the device and dataset in GLASStasks.py.

Though the seed has been fixed, the sparse matrix multiplication is not deterministic in PyTorch, it normal to observe some variance in performance.
