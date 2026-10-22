"""
Small 2D dataset for testing clustering algorithms
Creates 3 well-separated clusters in 2D space
"""

import numpy as np
import pandas as pd

def create_test_dataset():
    """Create a small 2D dataset with 3 clear clusters
    
    Returns:
        np.ndarray: 2D array with shape (15, 2) containing 3 clusters
        - Cluster 1: 5 points around (2, 2) - tight cluster
        - Cluster 2: 5 points around (8, 8) - tight cluster  
        - Cluster 3: 5 points around (2, 8) - tight cluster
    """
    np.random.seed(42)  # For reproducible results
    
    # Cluster 1: Bottom-left (tight cluster)
    cluster1 = np.random.normal([2, 2], 0.3, (5, 2))
    
    # Cluster 2: Top-right (tight cluster)
    cluster2 = np.random.normal([8, 8], 0.3, (5, 2))
    
    # Cluster 3: Top-left (tight cluster)
    cluster3 = np.random.normal([2, 8], 0.3, (5, 2))
    
    # Combine all clusters
    data = np.vstack([cluster1, cluster2, cluster3])
    
    return data

def create_test_dataset_with_outliers():
    """Create dataset with outliers to test robustness
    
    Returns:
        np.ndarray: 2D array with 3 clusters + 2 outliers
    """
    data = create_test_dataset()
    
    # Add outliers
    outliers = np.array([[10, 1], [1, 10]])
    data_with_outliers = np.vstack([data, outliers])
    
    return data_with_outliers

def create_overlapping_clusters():
    """Create overlapping clusters to test edge cases
    
    Returns:
        np.ndarray: 2D array with 2 overlapping clusters
    """
    np.random.seed(42)
    
    # Two clusters that overlap slightly
    cluster1 = np.random.normal([3, 3], 1.0, (5, 2))
    cluster2 = np.random.normal([5, 5], 1.0, (5, 2))
    
    return np.vstack([cluster1, cluster2])

def get_expected_labels():
    """Get expected cluster labels for the main test dataset
    
    Returns:
        list: Expected cluster assignment for each point
        [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
    """
    return [0]*5 + [1]*5 + [2]*5

def save_test_data():
    """Save test datasets to CSV files for easy testing"""
    # Main test dataset
    data = create_test_dataset()
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    df.to_csv('tests/clustering_test_data.csv', index=False)
    
    # Dataset with outliers
    data_outliers = create_test_dataset_with_outliers()
    df_outliers = pd.DataFrame(data_outliers, columns=['feature1', 'feature2'])
    df_outliers.to_csv('tests/clustering_test_data_outliers.csv', index=False)
    
    # Overlapping clusters
    data_overlap = create_overlapping_clusters()
    df_overlap = pd.DataFrame(data_overlap, columns=['feature1', 'feature2'])
    df_overlap.to_csv('tests/clustering_test_data_overlap.csv', index=False)

if __name__ == "__main__":
    save_test_data()
    print("Test datasets created successfully!")
    print("Files created:")
    print("- tests/clustering_test_data.csv (15 points, 3 clusters)")
    print("- tests/clustering_test_data_outliers.csv (17 points, 3 clusters + outliers)")
    print("- tests/clustering_test_data_overlap.csv (10 points, 2 overlapping clusters)")
