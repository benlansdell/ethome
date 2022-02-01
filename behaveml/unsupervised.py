from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
from scipy import signal
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import umap
from skimage.segmentation import watershed
from behaveml import VideosetDataFrame

def compute_tsne_embedding(dataset : VideosetDataFrame, 
                           cols : list, 
                           N_rows : int = 20000, 
                           n_components = 2, 
                           perplexity = 30) -> tuple:
    """Compute TSNE embedding. 
    
    Args:
        dataset: Input data
        cols: A list of column names to produce the embedding for
        N_rows: A number of rows to randomly sample for the embedding. Only these rows are embedded.
        
    Returns:
        The tuple: 
    """
    
    tsne_data = StandardScaler().fit_transform(dataset.data[cols])
    random_indices = np.random.choice(tsne_data.shape[0], min(N_rows, tsne_data.shape[0]), replace = False)
    tsne_data = tsne_data[random_indices, :]
    tsne_embedding = TSNE(n_components=n_components, init = 'pca', perplexity = perplexity).fit_transform(tsne_data)
    return tsne_embedding, random_indices

def compute_morlet(data : np.ndarray, 
                   dt : float = 1/30, 
                   n_freq : int = 5, 
                   w : float = 3) -> np.ndarray:
    fs = 1/dt
    freq = np.geomspace(1, fs/2, n_freq)
    widths = w*fs / (2*freq*np.pi)
    morlet = []
    for idx in tqdm(range(data.shape[1])):
        morlet.append(np.abs(signal.cwt(data[:,idx], signal.morlet2, widths, w=w)))
    morlet = np.stack(morlet)
    return morlet    

def compute_density(dataset : VideosetDataFrame, 
                    embedding_extent : tuple, 
                    bandwidth : float = 0.5, 
                    n_pts : int = 300) -> np.ndarray:
    """Compute kernel density estimate of embedding.
    
    Args:
        dataset: VideosetDataFrame with embedding data loaded in it. (Must have already populated columns named 'embedding_0', 'embedding_1')
        embedding_extent: the bounds in which to apply the density estimate
        bandwidth: the Gaussian kernel bandwidth. Will depend on the scale of the embedding. Can be changed to affect the number of clusters pulled out
        n_pts: number of points over which to evaluate the KDE

    Returns:
        Numpy array with KDE over the specified square region in the embedding space, with dimensions (n_pts x n_pts)    
    """

    xmin, xmax, ymin, ymax = embedding_extent
    den_est = KernelDensity(bandwidth = bandwidth)
    den_est.fit(dataset.data[['embedding_0', 'embedding_1']])
    X_plot = np.array(np.meshgrid(np.linspace(xmin, xmax, n_pts), np.linspace(ymin, ymax, n_pts))).T.reshape(-1, 2)
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

def cluster_behaviors(dataset, 
                      feature_cols, 
                      N_rows = 200000, 
                      subsample = True, 
                      use_morlet = False, 
                      use_umap = True, 
                      n_pts = 300, 
                      bandwidth = 0.5):
    """Cluster behaviors based on dimensionality reduction, kernel density estimation, and watershed clustering.
    
    """

    #If running interactively, can set default params here
    # N_rows = 200000
    # subsample = True 
    # use_morlet = False
    # use_umap = True
    # n_pts = 300
    # bandwidth = 0.5

    #TODO
    # Should be autodetected, separate for x and y
    xmin = -12
    xmax = 22

    ymin = -12
    ymax = 22

    embedding_extent = (xmin, xmax, ymin, ymax)

    data = dataset.data[feature_cols]

    if use_morlet:
        morlet = compute_morlet(data)
        morlet = morlet.reshape((morlet.shape[0]*morlet.shape[1], -1))
        morlet = morlet.T
        embedding_data = StandardScaler().fit_transform(morlet)
    else:
        embedding_data = StandardScaler().fit_transform(data)

    if subsample:
        random_indices = np.random.choice(embedding_data.shape[0], min(N_rows, embedding_data.shape[0]), replace = False)
        embedding_data_fit = embedding_data[random_indices, :]
    else:
        embedding_data_fit = embedding_data

    if use_umap:
        reducer = umap.UMAP()
        reducer.fit(embedding_data_fit)
        embedding = reducer.transform(embedding_data)
    else:
        embedding = TSNE(n_components=2, perplexity = 50, init = 'pca').fit_transform(embedding_data)

    dataset.data[['embedding_0', 'embedding_1']] = embedding

    dens_matrix = compute_density(dataset, embedding_extent, bandwidth = bandwidth, n_pts = n_pts)
    labels = compute_watershed(dens_matrix, positive_only = False, cutoff = 0)

    #Save data
    dataset.data['embedding_index_0'] = dataset.data['embedding_0'].apply(lambda x: int(max(0, min(n_pts-1, n_pts*(x-xmin)/(xmax-xmin)))))
    dataset.data['embedding_index_1'] = dataset.data['embedding_1'].apply(lambda x: int(max(0, min(n_pts-1, n_pts*(x-ymin)/(ymax-ymin)))))
    dataset.data['unsup_behavior_label'] = dataset.data[['embedding_index_0', 'embedding_index_1']].apply(lambda x: labels[x[1],x[0]], axis = 1)
    dataset.data.loc[dataset.data['unsup_behavior_label'] == 0, 'unsup_behavior_label'] = -1

    return (dens_matrix, labels, embedding_extent)