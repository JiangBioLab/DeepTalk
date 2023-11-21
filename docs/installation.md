# Installation
## How to install DeepTalk

To install DeepTalk, make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed. If you need more details on the dependences, look at the `environment.yml` file.

- set up conda environment for DeepTalk

```
	conda env create -f environment.yml
```

- install DeepTalk_ST from shell:

```
	conda activate deeptalk-env
	pip install DeepTalk_ST
```

- To start using DeepTalk, import DeepTalk in your jupyter notebooks or/and scripts

```
	import DeepTalk as dt
```

## How to run DeepTalk for cell type identification

Load your spatial data and your single cell data (which should be in [AnnData](https://anndata.readthedocs.io/en/latest/) format), and pre-process them using dt.pp_adatas`:

```
    ad_st = sc.read_h5ad(path)
    ad_sc = sc.read_h5ad(path)
    dt.pp_adatas(ad_sc, ad_st, genes=None)
```

The function `pp_adatas` finds the common genes between adata_sc, adata_sp, and saves them in two `adatas.uns` for mapping and analysis later. Also, it subsets the intersected genes to a set of training genes passed by `genes`. If `genes=None`, DeepTalk maps using all genes shared by the two datasets. Once the datasets are pre-processed we can map:

```
    ad_map = dt.map_cells_to_space(ad_sc, ad_st)
```

The returned AnnData,`ad_map`, is a cell-by-voxel structure where `ad_map.X[i, j]` gives the probability for cell `i` to be in voxel `j`. This structure can be used to project gene expression from the single cell data to space, which is achieved via `dt.project_genes`.

```
    ad_ge = dt.project_genes(ad_map, ad_sc)
```

The returned `ad_ge` is a voxel-by-gene AnnData, similar to spatial data `ad_st`, but where gene expression has been projected from the single cells. 

## How to run DeepTalk for cell-cell communication inference

Generating Feature Files for Deep Learning Using `ad_ge` :

```
    Fea_embeddings = dt.feature_pca(input_file, lrpairs,cell_pair_all)
```

Use subgraph-based graph attention network to construct CCC networks for the ligand-receptor pairs with a spatial distance constraint:

```
    cccpre = dt.CCC_Predict(data_name,
    data_path,outdir, Fea_embeddings)
```
