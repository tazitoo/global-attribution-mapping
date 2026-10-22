import numpy as np
import pytest

from gam.kendall_tau_distance import (ktau_weighted_distance,
                                      mergeSortDistance,
                                      pairwise_distance_matrix)

try:
    import dask.array as da
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def test_ktau_symmetry():
    """Checks symmetry holds for distance metric"""
    r1 = [0.1, 0.2, 0.7]
    assert ktau_weighted_distance(r1, r1) == 0


def test_ktau_relative_distances():
    """Compares relative distances of weighted rankings"""
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    # r2 and r3 should be closer than r2 to r1
    assert ktau_weighted_distance(r2, r3) < ktau_weighted_distance(r2, r1)


def test_pairwise_distance_matrix():
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    rankings = np.array([r1, r2, r3])
    D = pairwise_distance_matrix(rankings)

    # check symmetry, within floating point rounding margin
    assert (D[0][1] - D[1][0]) < 1e-9
    # check diagonal is zero
    assert D[1][1] == 0
    assert D[2][2] == 0
    # distance between r2 and r3 is closer than r2 and r1
    assert D[1][2] < D[1][0]


@pytest.mark.skipif(not DASK_AVAILABLE, reason="dask not installed")
def test_dask_pairwise_distance_matrix():
    client = Client()
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    rankings = np.array([r1, r2, r3])

    D = pairwise_distance_matrix(da.from_array(rankings))
    # check symmetry, within floating point rounding margin
    assert (D[0][1] - D[1][0]) < 1e-9
    # check diagonal is zero
    assert D[1][1] == 0
    assert D[2][2] == 0
    # distance between r2 and r3 is closer than r2 and r1
    assert D[1][2] < D[1][0]
    client.close()


def test_pairwise_distance_matrix_invalid_input():
    """Test that pairwise_distance_matrix raises TypeError for invalid input"""
    with pytest.raises(TypeError, match="rankings must be numpy.ndarray"):
        pairwise_distance_matrix([1, 2, 3])  # list instead of ndarray


def test_ktau_weighted_distance_edge_cases():
    """Test edge cases for ktau weighted distance"""
    # Test with zero weights
    r1 = [0.0, 0.0, 1.0]
    r2 = [0.0, 0.0, 1.0]
    assert ktau_weighted_distance(r1, r2) == 0
    
    # Test with different lengths should raise ValueError
    with pytest.raises(ValueError):
        ktau_weighted_distance([0.1, 0.9], [0.1, 0.8, 0.1])



def test_ktau_accuracy():
    """ Floating point accuracy test for testing faster calculation methods """
    r1 = [0.27, 0.24, 0.26, 0.23]
    r2 = [0.05, 0.2, 0.7, 0.05]
    assert ktau_weighted_distance(r1, r2) == 0.0031050000000000006
    assert mergeSortDistance(r1, r2) == 0.0031050000000000006
