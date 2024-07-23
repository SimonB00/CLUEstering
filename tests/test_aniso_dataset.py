'''
Testing the algorithm on the aniso dataset
'''

from check_result import check_result
import os
import sys
import pandas as pd
import pytest
sys.path.insert(1, '.')
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def aniso():
    '''
    Returns the dataframe containing the aniso ataset
    '''
    return pd.read_csv("./test_datasets/aniso_1000.csv")


def test_clustering(aniso):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./aniso_1000_output.csv'):
        os.remove('./aniso_1000_output.csv')

    c = clue.clusterer(28., 5., 1.)
    c.read_data(aniso)
    c.run_clue()
    c.to_csv('./', 'aniso_1000_output.csv')

    check_result('./aniso_1000_output.csv',
                 './test_datasets/truth_files/aniso_1000_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(28., 5., 1.)
    c.read_data("./test_datasets/aniso_1000.csv")
    c.run_clue()
    c.cluster_plotter()
