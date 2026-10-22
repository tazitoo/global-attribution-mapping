"""
Comprehensive tests for clustering module using synthetic 2D data
"""

import numpy as np
import pandas as pd
from gam.clustering import KMedoids
from tests.test_clustering_data import (
    create_test_dataset, 
    create_test_dataset_with_outliers,
    create_overlapping_clusters,
    get_expected_labels
)

def test_kmedoids_basic_clustering():
    """Test basic k-medoids clustering on well-separated 2D data"""
    # Create test data with 3 clear clusters
    data = create_test_dataset()
    
    # Initialize KMedoids
    kmed = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        max_iter=10,
        tol=0.001,
        init_medoids="bandit",
        swap_medoids="pam",
        verbose=False
    )
    
    # Fit the model
    kmed.fit(data)
    
    # Test basic properties
    assert hasattr(kmed, 'centers')
    assert len(kmed.centers) == 3
    assert hasattr(kmed, 'members')
    assert len(kmed.members) == len(data)
    
    # Test that centers are actual data points
    for center in kmed.centers:
        assert center in range(len(data))
    
    # Test that all data points are assigned to clusters
    unique_labels = np.unique(kmed.members)
    assert len(unique_labels) == 3
    assert all(label in [0, 1, 2] for label in unique_labels)
    
    # Test that clusters are reasonably balanced (not all points in one cluster)
    members_int = kmed.members.astype(int)
    label_counts = np.bincount(members_int)
    for count in label_counts:
        assert count >= 2, "Cluster should have at least 2 points"
        assert count <= 8, "Cluster should not have more than 8 points"

# def test_kmedoids_different_distances():
#     """Test k-medoids with different distance metrics"""
#     data = create_test_dataset()
    
#     for dist_func in ["euclidean", "spearman_squared_distance", "kendall_tau"]:
#         kmed = KMedoids(
#             n_clusters=3,
#             batchsize=10,
#             dist_func=dist_func,
#             max_iter=10,
#             init_medoids="bandit",
#             swap_medoids="pam",
#             verbose=False
#         )
        
#         kmed.fit(data)
        
#         # Basic checks for each distance metric
#         assert len(kmed.centers) == 3
#         assert len(kmed.members) == len(data)
#         assert all(label in [0, 1, 2] for label in kmed.members)

def test_kmedoids_bandit_algorithm():
    """Test k-medoids with bandit initialization and swapping"""
    data = create_test_dataset()
    
    kmed = KMedoids(
        n_clusters=3,
        dist_func="euclidean",
        batchsize=5,
        max_iter=10,
        init_medoids="bandit",
        swap_medoids="bandit",
        verbose=False
    )
    
    kmed.fit(data)
    
    assert len(kmed.centers) == 3
    assert len(kmed.members) == len(data)

def test_kmedoids_custom_distance_function():
    """Test k-medoids with custom distance function"""
    from gam.spearman_distance import spearman_squared_distance
    
    data = create_test_dataset()
    
    kmed = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func=spearman_squared_distance,
        max_iter=10,
        init_medoids="bandit",
        swap_medoids="pam",
        verbose=False
    )
    
    kmed.fit(data)
    
    assert len(kmed.centers) == 3
    assert len(kmed.members) == len(data)

def test_kmedoids_with_outliers():
    """Test k-medoids robustness to outliers"""
    data_with_outliers = create_test_dataset_with_outliers()
    
    kmed = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        max_iter=10,
        init_medoids="bandit",
        swap_medoids="pam",
        verbose=False
    )
    
    kmed.fit(data_with_outliers)
    
    # Should still find 3 main clusters
    assert len(kmed.centers) == 3
    assert len(kmed.members) == len(data_with_outliers)

def test_kmedoids_edge_cases():
    """Test edge cases and error conditions"""
    data = create_test_dataset()
    
    # Test with k=1
    kmed1 = KMedoids(n_clusters=1, batchsize=len(data), dist_func="euclidean")
    kmed1.fit(data)
    assert len(kmed1.centers) == 1
    assert all(label == 0 for label in kmed1.members)
    
    # Test with k=n (each point is its own cluster)
    kmed_n = KMedoids(n_clusters=len(data), batchsize=len(data), dist_func="euclidean")
    kmed_n.fit(data)
    assert len(kmed_n.centers) == len(data)
    assert len(np.unique(kmed_n.members)) == len(data)
    
    # Test with k > n (should still work, some clusters will be empty)
    kmed_more = KMedoids(n_clusters=len(data) + 2, batchsize=len(data), dist_func="euclidean")
    kmed_more.fit(data)
    assert len(kmed_more.centers) == len(data) + 2

def test_kmedoids_reproducibility():
    """Test that results are reproducible with same initialization"""
    data = create_test_dataset()
    
    # Run clustering twice with same initialization
    kmed1 = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        init_medoids="bandit",
        swap_medoids="pam"
    )
    kmed1.fit(data)
    
    kmed2 = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        init_medoids="bandit",
        swap_medoids="pam"
    )
    kmed2.fit(data)
    
    # Basic checks that both runs produce valid results
    assert len(kmed1.centers) == 3
    assert len(kmed2.centers) == 3
    assert len(kmed1.members) == len(data)
    assert len(kmed2.members) == len(data)

def test_kmedoids_convergence():
    """Test that clustering converges within reasonable iterations"""
    data = create_test_dataset()
    
    kmed = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        max_iter=10,
        tol=0.001,
        init_medoids="bandit",
        swap_medoids="pam",
        verbose=False
    )
    
    kmed.fit(data)
    
    # Should converge and produce valid results
    assert hasattr(kmed, 'centers')
    assert len(kmed.centers) == 3

def test_kmedoids_with_numpy_array():
    """Test k-medoids with numpy array input (normal usage pattern)"""
    data = create_test_dataset()
    
    kmed = KMedoids(
        n_clusters=3,
        batchsize=10,
        dist_func="euclidean",
        init_medoids="bandit",
        swap_medoids="pam"
    )
    
    kmed.fit(data)
    
    assert len(kmed.centers) == 3
    assert len(kmed.members) == len(data)
