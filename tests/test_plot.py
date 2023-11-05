#####################
## Setup test code ##
#####################

import pandas as pd

def test_umap_plotting(dataset, default_track_cols):

    from ethome.unsupervised import compute_umap_embedding
    from ethome.plot import plot_embedding
    import matplotlib

    N_ROWS = 1000
    embedding = compute_umap_embedding(dataset, default_track_cols, N_rows = N_ROWS)
    dataset[['embedding_0', 'embedding_1']] = embedding
    fig, ax = plot_embedding(dataset) 

    assert type(fig) is matplotlib.figure.Figure 