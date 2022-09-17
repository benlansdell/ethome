<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `unsupervised`





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_tsne_embedding`

```python
compute_tsne_embedding(
    dataset: ExperimentDataFrame,
    cols: list,
    N_rows: int = 20000,
    n_components=2,
    perplexity=30
) → tuple
```

Compute TSNE embedding. Only for a random subset of rows. 



**Args:**
 
 - <b>`dataset`</b>:  Input data 
 - <b>`cols`</b>:  A list of column names to produce the embedding for 
 - <b>`N_rows`</b>:  A number of rows to randomly sample for the embedding. Only these rows are embedded. 
 - <b>`n_components`</b>:  The number of dimensions to embed the data into. 
 - <b>`perplexity`</b>:  The perplexity of the TSNE embedding. 



**Returns:**
 The tuple:  
        - A numpy array with the embedding data, only for a random subset of row 
        - The rows that were used for the embedding 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_morlet`

```python
compute_morlet(
    data: ndarray,
    dt: float = 0.03333333333333333,
    n_freq: int = 5,
    w: float = 3
) → ndarray
```

Compute morlet wavelet transform of a time series. 



**Args:**
 
 - <b>`data`</b>:  A 2D array containing the time series data, with dimensions (n_pts x n_channels) 
 - <b>`dt`</b>:  The time step of the time series 
 - <b>`n_freq`</b>:  The number of frequencies to compute 
 - <b>`w`</b>:  The width of the morlet wavelet 

Returns A 2D numpy array with the morlet wavelet transform. The first dimension is the frequency, the second is the time. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_density`

```python
compute_density(
    dataset: ExperimentDataFrame,
    embedding_extent: tuple,
    bandwidth: float = 0.5,
    n_pts: int = 300,
    N_sample_rows: int = 50000,
    rows: list = None
) → ndarray
```

Compute kernel density estimate of embedding. 



**Args:**
 
 - <b>`dataset`</b>:  ExperimentDataFrame with embedding data loaded in it. (Must have already populated columns named 'embedding_0', 'embedding_1') 
 - <b>`embedding_extent`</b>:  the bounds in which to apply the density estimate. Has the form (xmin, xmax, ymin, ymax) 
 - <b>`bandwidth`</b>:  the Gaussian kernel bandwidth. Will depend on the scale of the embedding. Can be changed to affect the number of clusters pulled out 
 - <b>`n_pts`</b>:  number of points over which to evaluate the KDE 
 - <b>`N_sample_rows`</b>:  number of rows to randomly sample to generate estimate 
 - <b>`rows`</b>:  If provided, use these rows instead of a random sample 



**Returns:**
 Numpy array with KDE over the specified square region in the embedding space, with dimensions (n_pts x n_pts)     


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_watershed`

```python
compute_watershed(
    dens_matrix: ndarray,
    positive_only: bool = False,
    cutoff: float = 0
) → tuple
```

Compute watershed clustering of a density matrix.  



**Args:**
 
 - <b>`dens_matrix`</b>:  A square 2D numpy array, output from compute_density, containing the kernel density estimate of the embedding. 
 - <b>`positive_only`</b>:  Whether to apply a threshold, 'cutoff'. If applied, 'cutoff' is subtracted from dens_matrix, and any value below zero is set to zero. Useful for only focusing on high density clusters. 
 - <b>`cutoff`</b>:  The cutoff value to apply if positive_only = True 



**Returns:**
 A numpy array with the same dimensions as dens_matrix. Each value in the array is the cluster ID for that coordinate. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/unsupervised.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cluster_behaviors`

```python
cluster_behaviors(
    dataset: ExperimentDataFrame,
    feature_cols: list,
    N_rows: int = 200000,
    use_morlet: bool = False,
    use_umap: bool = True,
    n_pts: int = 300,
    bandwidth: float = 0.5,
    **kwargs
) → tuple
```

Cluster behaviors based on dimensionality reduction, kernel density estimation, and watershed clustering. 

**Note that this will modify the dataset dataframe in place.** 

The following columns are added to dataset:   'embedding_index_[0/1]': the coordinates of each embedding coordinate in the returned density matrix  'unsup_behavior_label': the Watershed transform label for that row, based on its embedding coordinates. Rows whose embedding coordinate has no watershed cluster, or which fall outside the domain have value -1. 



**Args:**
 
 - <b>`dataset`</b>:  the ExperimentDataFrame with the features of interest 
 - <b>`feature_cols`</b>:  list of column names to perform the clustering on 
 - <b>`N_rows`</b>:  number of rows to perform the embedding on. If 'None', then all rows are used. 
 - <b>`use_morlet`</b>:  Apply Morlet wavelet transform to the feature cols before computing the embedding 
 - <b>`use_umap`</b>:  If True will use UMAP dimensionality reduction, if False will use TSNE 
 - <b>`n_pts`</b>:  dimension of grid the kernel density estimate is evaluated on.  
 - <b>`bandwidth`</b>:  Gaussian kernel bandwidth for kernel estimate 
 - <b>`**kwargs`</b>:  All other keyword parameters are sent to dimensionality reduction call (either TSNE or UMAP) 



**Returns:**
 A tuple with components: 
 - <b>`dens_matrix`</b>:  the (n_pts x n_pts) numpy array with the density estimate of the 2D embedding 
 - <b>`labels`</b>:  numpy array with same dimensions are dens_matrix, but with values the watershed cluster IDs 
 - <b>`embedding_extent`</b>:  the coordinates in embedding space that dens_matrix is approximating the density over 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
