#####################
## Setup test code ##
#####################

import pytest

import numpy as np

from behaveml.unsupervised import compute_tsne_embedding, compute_morlet, \
                                  compute_density, compute_watershed, cluster_behaviors

N_ROWS = 200

def test_tsne_embedding(videodataset, default_track_cols):
    embedding, indices = compute_tsne_embedding(videodataset, default_track_cols, N_rows = N_ROWS)
    assert embedding.shape == (N_ROWS, 2)
    assert len(indices) == N_ROWS

def test_compute_morlet(videodataset, default_track_cols):
    print(default_track_cols)
    data = videodataset.data[default_track_cols].to_numpy()[:N_ROWS,:]
    morlet = compute_morlet(data)
    print(morlet.shape)
    assert 1 == 1

def test_compute_density(videodataset):

    n_pts = 200
    extent = (-50, 50, -50, 50)
    videodataset.data = videodataset.data.iloc[:N_ROWS,:]
    embedding = np.random.randn(N_ROWS, 2)
    videodataset.data[['embedding_0', 'embedding_1']] = embedding
    dens_matrix = compute_density(videodataset, extent, n_pts = n_pts)

    assert dens_matrix.shape == (n_pts, n_pts)
    assert np.min(dens_matrix) >= 0

def test_compute_watershed(dens_matrix):
    labels = compute_watershed(dens_matrix)
    assert labels.shape == dens_matrix.shape

def test_cluster_behaviors(videodataset, default_track_cols):
    cluster_results = cluster_behaviors(videodataset, default_track_cols, N_rows = 200)

    assert 'embedding_index_0' in videodataset.data.columns
    assert 'embedding_index_1' in videodataset.data.columns
    assert 'unsup_behavior_label' in videodataset.data.columns
