from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np

def compute_tsne_embedding(data, cols, N_rows = 20000):
    tsne_data = StandardScaler().fit_transform(data[cols])
    random_indices = np.random.choice(tsne_data.shape[0], N_rows, replace = False)
    tsne_data = tsne_data[random_indices, :]
    tsne_embedding = TSNE(n_components=2, init = 'pca').fit_transform(tsne_data)
    return tsne_embedding, random_indices