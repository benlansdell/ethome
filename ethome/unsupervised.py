from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
from scipy import signal
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import umap
from skimage.segmentation import watershed
import pandas as pd

def compute_tsne_embedding(dataset : pd.DataFrame, 
                           cols : list, 
                           N_rows : int = 20000, 
                           n_components = 2, 
                           perplexity = 30) -> tuple:
    """Compute TSNE embedding. Only for a random subset of rows.
    
    Args:
        dataset: Input data
        cols: A list of column names to produce the embedding for
        N_rows: A number of rows to randomly sample for the embedding. Only these rows are embedded.
        n_components: The number of dimensions to embed the data into.
        perplexity: The perplexity of the TSNE embedding.

    Returns:
        The tuple: 
            - A numpy array with the embedding data, only for a random subset of row
            - The rows that were used for the embedding
    """
    
    tsne_data = StandardScaler().fit_transform(dataset[cols])
    random_indices = np.random.choice(tsne_data.shape[0], min(N_rows, tsne_data.shape[0]), replace = False)
    tsne_data = tsne_data[random_indices, :]
    tsne_embedding = TSNE(n_components=n_components, init = 'pca', perplexity = perplexity).fit_transform(tsne_data)
    return tsne_embedding, random_indices

def compute_umap_embedding(dataset : pd.DataFrame, 
                           cols : list, 
                           N_rows : int = 20000, 
                           n_components = 2, 
                           **kwargs) -> np.ndarray:
    """Compute UMAP embedding. Compute based on a random subset of rows, then apply to all rows
    
    Args:
        dataset: Input data
        cols: A list of column names to produce the embedding for
        N_rows: A number of rows to randomly sample for the embedding. Only these rows are embedded.
        n_components: The number of dimensions to embed the data into.
        **kwargs: Passed to UMAP constructor
        
    Returns:
        - A numpy array with the embedding data
    """
    
    umap_data = StandardScaler().fit_transform(dataset[cols])
    random_indices = np.random.choice(umap_data.shape[0], min(N_rows, umap_data.shape[0]), replace = False)
    umap_data_rand = umap_data[random_indices, :]
    model = umap.UMAP(n_components=n_components, **kwargs)
    model.fit(umap_data_rand)
    return model.transform(umap_data)

def compute_morlet(data : np.ndarray, 
                   dt : float = 1/30, 
                   n_freq : int = 5, 
                   w : float = 3) -> np.ndarray:
    """ Compute morlet wavelet transform of a time series.
    
    Args:
        data: A 2D array containing the time series data, with dimensions (n_pts x n_channels)
        dt: The time step of the time series
        n_freq: The number of frequencies to compute
        w: The width of the morlet wavelet
        
    Returns
        A 2D numpy array with the morlet wavelet transform. The first dimension is the frequency, the second is the time.
    """
    if type(data) is pd.DataFrame:
        data = data.to_numpy()
    fs = 1/dt
    freq = np.geomspace(1, fs/2, n_freq)
    widths = w*fs / (2*freq*np.pi)
    morlet = []
    for idx in tqdm(range(data.shape[1])):
        morlet.append(np.abs(signal.cwt(data[:,idx], signal.morlet2, widths, w=w)))
    morlet = np.stack(morlet)
    return morlet    

def compute_density(dataset : pd.DataFrame, 
                    embedding_extent : tuple, 
                    bandwidth : float = 0.5, 
                    n_pts : int = 300,
                    N_sample_rows : int = 50000,
                    rows : list = None) -> np.ndarray:
    """Compute kernel density estimate of embedding.
    
    Args:
        dataset: pd.DataFrame with embedding data loaded in it. (Must have already populated columns named 'embedding_0', 'embedding_1')
        embedding_extent: the bounds in which to apply the density estimate. Has the form (xmin, xmax, ymin, ymax)
        bandwidth: the Gaussian kernel bandwidth. Will depend on the scale of the embedding. Can be changed to affect the number of clusters pulled out
        n_pts: number of points over which to evaluate the KDE
        N_sample_rows: number of rows to randomly sample to generate estimate
        rows: If provided, use these rows instead of a random sample

    Returns:
        Numpy array with KDE over the specified square region in the embedding space, with dimensions (n_pts x n_pts)    
    """

    xmin, xmax, ymin, ymax = embedding_extent
    den_est = KernelDensity(bandwidth = bandwidth)
    if rows is not None:
        sample_rows = rows
    else:
        sample_rows = np.random.choice(dataset.index, min(N_sample_rows, len(dataset)), replace = False)
    print("Fitting KDE")
    den_est.fit(dataset.loc[sample_rows, ['embedding_0', 'embedding_1']])
    X_plot = np.array(np.meshgrid(np.linspace(xmin, xmax, n_pts), np.linspace(ymin, ymax, n_pts))).T.reshape(-1, 2)
    print("Computing grid of densities")
    dens = den_est.score_samples(X_plot)
    dens = np.exp(dens)
    dens_matrix = dens.reshape(n_pts, n_pts).T
    return dens_matrix

def compute_watershed(dens_matrix : np.ndarray, 
                      positive_only : bool = False, 
                      cutoff : float = 0) -> tuple:
    """Compute watershed clustering of a density matrix. 
    
    Args:
        dens_matrix: A square 2D numpy array, output from compute_density, containing the kernel density estimate of the embedding.
        positive_only: Whether to apply a threshold, 'cutoff'. If applied, 'cutoff' is subtracted from dens_matrix, and any value below zero is set to zero. Useful for only focusing on high density clusters.
        cutoff: The cutoff value to apply if positive_only = True
        
    Returns:
        A numpy array with the same dimensions as dens_matrix. Each value in the array is the cluster ID for that coordinate.
    """

    if positive_only:
        dm = np.maximum(cutoff, dens_matrix)-cutoff
    else:
        dm = dens_matrix
    image = (255*dm/np.max(dm)).astype(int)
    labels = watershed(-dm, None, mask = image>0)
    return labels

def cluster_behaviors(dataset : pd.DataFrame, 
                      feature_cols : list, 
                      N_rows : int = 200000, 
                      use_morlet : bool = False, 
                      use_umap :bool = True, 
                      n_pts : int = 300, 
                      bandwidth : float = 0.5,
                      **kwargs) -> tuple:
    """Cluster behaviors based on dimensionality reduction, kernel density estimation, and watershed clustering.
    
    **Note that this will modify the dataset dataframe in place.**

    The following columns are added to dataset: 
        'embedding_index_[0/1]': the coordinates of each embedding coordinate in the returned density matrix
        'unsup_behavior_label': the Watershed transform label for that row, based on its embedding coordinates. Rows whose embedding coordinate has no watershed cluster, or which fall outside the domain have value -1.

    Args:
        dataset: the pd.DataFrame with the features of interest
        feature_cols: list of column names to perform the clustering on
        N_rows: number of rows to perform the embedding on. If 'None', then all rows are used.
        use_morlet: Apply Morlet wavelet transform to the feature cols before computing the embedding
        use_umap: If True will use UMAP dimensionality reduction, if False will use TSNE
        n_pts: dimension of grid the kernel density estimate is evaluated on. 
        bandwidth: Gaussian kernel bandwidth for kernel estimate
        **kwargs: All other keyword parameters are sent to dimensionality reduction call (either TSNE or UMAP)

    Returns:
        A tuple with components:
            dens_matrix: the (n_pts x n_pts) numpy array with the density estimate of the 2D embedding
            labels: numpy array with same dimensions are dens_matrix, but with values the watershed cluster IDs
            embedding_extent: the coordinates in embedding space that dens_matrix is approximating the density over
    """

    #If running interactively, can set default params here
    # N_rows = 200000
    # subsample = True 
    # use_morlet = False
    # use_umap = True
    # n_pts = 300
    # bandwidth = 0.5
    # kwargs = {}

    data = dataset[feature_cols]

    if use_morlet:
        print("Computing morlet transform of features")
        morlet = compute_morlet(data)
        morlet = morlet.reshape((morlet.shape[0]*morlet.shape[1], -1))
        morlet = morlet.T
        embedding_data = StandardScaler().fit_transform(morlet)
    else:
        embedding_data = StandardScaler().fit_transform(data)

    if N_rows is not None:
        random_indices = np.random.choice(embedding_data.shape[0], min(N_rows, embedding_data.shape[0]), replace = False)
        embedding_data_fit = embedding_data[random_indices, :]
    else:
        embedding_data_fit = embedding_data

    print("Performing dimensionality reduction")
    if use_umap:
        reducer = umap.UMAP(**kwargs)
        reducer.fit(embedding_data_fit)
        embedding = reducer.transform(embedding_data)
    else:
        if 'perplexity' not in kwargs:
            kwargs['perplexity'] = 50
        if 'init' not in kwargs:
            kwargs['init'] = 'pca'
        embedding = TSNE(n_components=2, **kwargs).fit_transform(embedding_data)

    dataset[['embedding_0', 'embedding_1']] = embedding

    xmin = np.quantile(dataset['embedding_0'], 0.01)
    xmax = np.quantile(dataset['embedding_0'], 0.99)
    ymin = np.quantile(dataset['embedding_1'], 0.01)
    ymax = np.quantile(dataset['embedding_1'], 0.99)

    xrange = (xmax - xmin)
    yrange = (ymax - ymin)
    xmin -= 0.1*xrange 
    xmax += 0.1*xrange
    ymin -= 0.1*yrange
    ymax += 0.1*yrange

    embedding_extent = (xmin, xmax, ymin, ymax)

    print("Computing density estimate of embedding")
    dens_matrix = compute_density(dataset, embedding_extent, bandwidth = bandwidth, n_pts = n_pts)
    print("Performing watershed transform of density")
    labels = compute_watershed(dens_matrix, positive_only = False, cutoff = 0)

    dataset['embedding_index_0'] = dataset['embedding_0'].apply(lambda x: int(max(0, min(n_pts-1, n_pts*(x-xmin)/(xmax-xmin)))))
    dataset['embedding_index_1'] = dataset['embedding_1'].apply(lambda x: int(max(0, min(n_pts-1, n_pts*(x-ymin)/(ymax-ymin)))))
    dataset['unsup_behavior_label'] = dataset[['embedding_index_0', 'embedding_index_1']].apply(lambda x: labels[x[1],x[0]], axis = 1)
    dataset.loc[dataset['unsup_behavior_label'] == 0, 'unsup_behavior_label'] = -1

    return (dens_matrix, labels, embedding_extent)