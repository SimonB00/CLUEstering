'''
Testing the algorithm on the circle dataset, a dataset where points are
distributed to simulate the hits of a small set of particles in a detector.
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def toy_det_1000():
    '''
    Returns the dataframe containing the toy-detector dataset
    with 1000 points.
    '''
    return pd.read_csv("./test_datasets/toyDetector_1000.csv")


@pytest.fixture
def toy_det_5000():
    '''
    Returns the dataframe containing the toy-detector dataset
    with 5000 points.
    '''
    return pd.read_csv("./test_datasets/toyDetector_1000.csv")


@pytest.fixture
def toy_det_10000():
    '''
    Returns the dataframe containing the toy-detector dataset
    with 10000 points.
    '''
    return pd.read_csv("./test_datasets/toyDetector_1000.csv")


def test_toy_1000(toy_det_1000):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./toy_det_1000_output.csv'):
        os.remove('./toy_det_1000_output.csv')

    c = clue.clusterer(5., 2.5, 1.)
    c.read_data(toy_det_1000)
    c.run_clue()
    c.to_csv('./', 'toy_det_1000_output.csv')

    check_result('./toy_det_1000_output.csv',
                 './test_datasets/truth_files/toy_det_1000_truth.csv')


def test_toy_5000(toy_det_5000):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./toy_det_5000_output.csv'):
        os.remove('./toy_det_5000_output.csv')

    c = clue.clusterer(5., 2.5, 1.)
    c.read_data(toy_det_5000)
    c.run_clue()
    c.to_csv('./', 'toy_det_5000_output.csv')

    check_result('./toy_det_5000_output.csv',
                 './test_datasets/truth_files/toy_det_5000_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(5., 2.5, 1.)
    c.read_data("./test_datasets/toyDetector_10000.csv")
    c.input_plotter()
    c.run_clue()
    print(c.n_clusters)
    c.cluster_plotter()
    # c.to_csv('./test_datasets/truth_files/', 'toy_det_5000_output.csv')
