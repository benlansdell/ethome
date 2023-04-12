#####################
## Setup test code ##
#####################

import pytest

import numpy as np

from ethome.unsupervised import compute_tsne_embedding, compute_morlet, compute_umap_embedding, \
                                  compute_density, compute_watershed, cluster_behaviors

N_ROWS = 200

def test_umap_embedding(dataset, default_track_cols):
    embedding = compute_umap_embedding(dataset, default_track_cols, N_rows = N_ROWS)
    assert embedding.shape == (len(dataset), 2)

def test_tsne_embedding(dataset, default_track_cols):
    embedding, indices = compute_tsne_embedding(dataset, default_track_cols, N_rows = N_ROWS)
    assert embedding.shape == (N_ROWS, 2)
    assert len(indices) == N_ROWS

def test_compute_morlet(dataset, default_track_cols):
    print(default_track_cols)
    data = dataset[default_track_cols].to_numpy()[:N_ROWS,:]
    morlet = compute_morlet(data)
    print(morlet.shape)
    assert 1 == 1

def test_compute_density(dataset):

    n_pts = 200
    extent = (-50, 50, -50, 50)
    dataset = dataset.iloc[:N_ROWS,:]
    embedding = np.random.randn(N_ROWS, 2)
    dataset[['embedding_0', 'embedding_1']] = embedding
    dens_matrix = compute_density(dataset, extent, n_pts = n_pts)

    assert dens_matrix.shape == (n_pts, n_pts)
    assert np.min(dens_matrix) >= 0

def test_compute_watershed(dens_matrix):
    labels = compute_watershed(dens_matrix)
    assert labels.shape == dens_matrix.shape

def test_cluster_behaviors(dataset, default_track_cols):
    cluster_results = cluster_behaviors(dataset, default_track_cols, N_rows = 200)
    assert 'embedding_index_0' in dataset.columns
    assert 'embedding_index_1' in dataset.columns
    assert 'unsup_behavior_label' in dataset.columns

    #Test Morlet
    #cluster_results = cluster_behaviors(dataset, default_track_cols, use_morlet = True, N_rows = 200)

    #Test TSNE
    #cluster_results = cluster_behaviors(dataset, default_track_cols, use_umap = False, N_rows = 200)
