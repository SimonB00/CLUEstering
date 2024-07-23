'''
Testing the algorithm on the blob dataset, a dataset where points are distributed to form
round clusters
'''

import os
import sys
import pandas as pd
import pytest
sys.path.insert(1, '.')
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blobs.csv")


@pytest.fixture
def blobs_noise():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blobs_noise.csv")


@pytest.fixture
def blobs_3d():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blobs_3d.csv")


@pytest.fixture
def blobs_3d_noise():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blob_3d_noise.csv")


def test_blobs(blobs):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_output.csv'):
        os.remove('./blobs_output.csv')

    c = clue.clusterer(1., 5., 1.5)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./', 'blobs_output.csv')

    check_result('./blobs_output.csv',
                 './test_datasets/truth_files/blobs_truth.csv')


def test_blobs_noise(blobs_noise):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_noise_output.csv'):
        os.remove('./blobs_noise_output.csv')

    c = clue.clusterer(1., 5., 1.5)
    c.read_data(blobs_noise)
    c.run_clue()
    c.to_csv('./', 'blobs_noise_output.csv')

    check_result('./blobs_noise_output.csv',
                 './test_datasets/truth_files/blobs_noise_truth.csv')


def test_blobs_3d(blobs_3d):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_3d_output.csv'):
        os.remove('./blobs_3d_output.csv')

    c = clue.clusterer(1., 5., 1.5)
    c.read_data(blobs_3d)
    c.run_clue()
    c.to_csv('./', 'blobs_3d_output.csv')

    check_result('./blobs_3d_output.csv',
                 './test_datasets/truth_files/blobs_3d_truth.csv')


def test_blobs_3d_noise(blobs_3d_noise):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_3d_noise_output.csv'):
        os.remove('./blobs_3d_noise_output.csv')

    c = clue.clusterer(1., 5., 1.5)
    c.read_data(blobs_3d_noise)
    c.run_clue()
    c.to_csv('./', 'blobs_3d_noise_output.csv')

    check_result('./blobs_3d_noise_output.csv',
                 './test_datasets/truth_files/blobs_3d_noise_truth.csv')


if __name__ == "__main__":
    c1 = clue.clusterer(1., 5, 1.5)
    c1.read_data("./test_datasets/blobs.csv")
    c1.run_clue()
    c1.cluster_plotter()

    c2 = clue.clusterer(1., 5, 1.5)
    c2.read_data("./test_datasets/blobs_noise.csv")
    c2.run_clue()
    c2.cluster_plotter()

    c3 = clue.clusterer(1., 5, 1.5)
    c3.read_data("./test_datasets/blobs_3d.csv")
    c3.run_clue()
    c3.cluster_plotter()

    c4 = clue.clusterer(1., 5, 1.5)
    c4.read_data("./test_datasets/blobs_3d_noise.csv")
    c4.run_clue()
    c4.cluster_plotter()
