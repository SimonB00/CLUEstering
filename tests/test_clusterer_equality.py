import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def moons():
    return pd.read_csv("./test_datasets/moons_1000.csv")

@pytest.fixture
def circles():
    return pd.read_csv("./test_datasets/circles_1000.csv")


def test_clusterer_equality(moons, circles):
    # Moons dataset
    clust1 = clue.clusterer(0.5, 5, 1.)
    clust1.read_data(moons)
    clust1.run_clue()
    clust1.cluster_plotter()

    # Create a copy of the moons clusterer to check the equality of clusterers
    clust1_copy = clue.clusterer(0.5, 5, 1.)
    clust1_copy.read_data(moons)
    clust1_copy.run_clue()
    clust1_copy.cluster_plotter()

    # Circles dataset
    clust2 = clue.clusterer(0.9,5,1.5)
    clust2.read_data(circles)
    clust2.run_clue()
    clust2.cluster_plotter()

    # Create a copy of the circles clusterer to check the equality of clusterers
    clust2_copy = clue.clusterer(0.9,5,1.5)
    clust2_copy.read_data(circles)
    clust2_copy.run_clue()
    clust2_copy.cluster_plotter()

    # Check equality
    assert clust1.clust_prop == clust1_copy.clust_prop
    assert clust2.clust_prop == clust2_copy.clust_prop

    # Check inequality
    assert clust1.clust_prop != clust2.clust_prop


if __name__ == "__main__":
    # Moons dataset
    clust1 = clue.clusterer(0.5, 5, 1.)
    clust1.read_data('./test_datasets/moons_1000.csv')
    clust1.run_clue()
    clust1.cluster_plotter()

    # Create a copy of the moons clusterer to check the equality of clusterers
    clust1_copy = clue.clusterer(0.5, 5, 1.)
    clust1_copy.read_data('./test_datasets/moons_1000.csv')
    clust1_copy.run_clue()
    clust1_copy.cluster_plotter()

    # Circles dataset
    clust2 = clue.clusterer(0.9,5,1.5)
    clust2.read_data('./test_datasets/circles_1000.csv')
    clust2.run_clue()
    clust2.cluster_plotter()

    # Create a copy of the circles clusterer to check the equality of clusterers
    clust2_copy = clue.clusterer(0.9,5,1.5)
    clust2_copy.read_data('./test_datasets/circles_1000.csv')
    clust2_copy.run_clue()
    clust2_copy.cluster_plotter()

    # Check equality
    assert clust1.clust_prop == clust1_copy.clust_prop
    assert clust2.clust_prop == clust2_copy.clust_prop

    # Check inequality
    assert clust1.clust_prop != clust2.clust_prop