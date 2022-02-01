<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `unsupervised`





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_tsne_embedding`

```python
compute_tsne_embedding(
    dataset: VideosetDataFrame,
    cols: list,
    N_rows: int = 20000,
    n_components=2,
    perplexity=30
) → tuple
```

Compute TSNE embedding.  



**Args:**
 
 - <b>`dataset`</b>:  Input data 
 - <b>`cols`</b>:  A list of column names to produce the embedding for 
 - <b>`N_rows`</b>:  A number of rows to randomly sample for the embedding. Only these rows are embedded. 



**Returns:**
 The tuple:  


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_morlet`

```python
compute_morlet(
    data: ndarray,
    dt: float = 0.03333333333333333,
    n_freq: int = 5,
    w: float = 3
) → ndarray
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_density`

```python
compute_density(
    dataset: VideosetDataFrame,
    embedding_extent: tuple,
    bandwidth: float = 0.5,
    n_pts: int = 300
) → ndarray
```

Compute kernel density estimate of embedding. 



**Args:**
 
 - <b>`dataset`</b>:  VideosetDataFrame with embedding data loaded in it. (Must have already populated columns named 'embedding_0', 'embedding_1') 
 - <b>`embedding_extent`</b>:  the bounds in which to apply the density estimate 
 - <b>`bandwidth`</b>:  the Gaussian kernel bandwidth. Will depend on the scale of the embedding. Can be changed to affect the number of clusters pulled out 
 - <b>`n_pts`</b>:  number of points over which to evaluate the KDE 



**Returns:**
 Numpy array with KDE over the specified square region in the embedding space, with dimensions (n_pts x n_pts)     


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/unsupervised.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cluster_behaviors`

```python
cluster_behaviors(
    dataset,
    feature_cols,
    N_rows=200000,
    subsample=True,
    use_morlet=False,
    use_umap=True,
    n_pts=300,
    bandwidth=0.5
)
```

Cluster behaviors based on dimensionality reduction, kernel density estimation, and watershed clustering.  

 






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
