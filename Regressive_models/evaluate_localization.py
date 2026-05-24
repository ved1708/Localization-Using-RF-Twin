#!/usr/bin/env python3
"""
Indoor Localization using RF Fingerprinting.

This script implements a kNN(3, 5, 10) based localization algorithm:
1. Loads the RF fingerprint dataset
2. Splits into training and test sets
3. For each test sample, finds k nearest neighbors based on RF feature similarity
4. Estimates position as weighted average of neighbor positions
5. Evaluates performance
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def load_dataset(filepath):
    """Load the RF fingerprint dataset."""
    print(f"Loading dataset from: {filepath}")
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"✓ Loaded {len(dataset)} samples")
    return dataset

def create_feature_vector(features):
    """Convert RF features dictionary to a single feature vector."""
    # Get path gains and delays
    gains = features['path_gains']
    delays = features['path_delays']
    
    # Take top 10 strongest paths
    n_paths = 10
    top_indices = np.argsort(gains)[::-1][:n_paths]
    
    gains_vec = np.zeros(n_paths)
    delays_vec = np.zeros(n_paths)
    
    gains_vec[:len(top_indices)] = gains[top_indices]
    delays_vec[:len(top_indices)] = delays[top_indices]
    
    # Delay normalization
    delays_vec = delays_vec * 1e6
    
    # Concatenate all features
    feature_vec = np.concatenate([
        gains_vec,
        delays_vec,
        [features['total_power']],
        [features['num_paths']],
        [features['delay_spread'] * 1e6]  # to microseconds
    ])
    
    return feature_vec

def prepare_data(dataset, test_split=0.2):
    """    
    Args:
        dataset: List of data points
        test_split: Fraction of data to use for testing
    
    Returns:
        train_positions, train_features, test_positions, test_features
    """
    n_samples = len(dataset)
    n_test = int(n_samples * test_split)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Extract positions and features
    train_positions = np.array([dataset[i]['position'] for i in train_indices])
    test_positions = np.array([dataset[i]['position'] for i in test_indices])
    
    train_features = np.array([create_feature_vector(dataset[i]['features']) for i in train_indices])
    test_features = np.array([create_feature_vector(dataset[i]['features']) for i in test_indices])
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Test samples: {len(test_indices)}")
    
    return train_positions, train_features, test_positions, test_features

def knn_localization(train_positions, train_features, test_features, k=5):
    """
    Args:
        train_positions: Nx3 array of known positions
        train_features: NxF array of RF features at those positions
        test_features: MxF array of RF features to localize
        k: Number of nearest neighbors
    
    Returns:
        predicted_positions: Mx3 array of estimated positions
    """
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(train_features)
    distances, indices = nbrs.kneighbors(test_features)
    
    # Estimate position as weighted average of k nearest neighbors
    weights = 1.0 / (distances + 1e-6)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    predicted_positions = np.zeros((len(test_features), 3))
    for i in range(len(test_features)):
        neighbor_positions = train_positions[indices[i]]
        predicted_positions[i] = np.sum(neighbor_positions * weights[i, :, np.newaxis], axis=0)
    
    return predicted_positions

def evaluate_performance(true_positions, predicted_positions):
    """    
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Compute position errors
    errors = np.linalg.norm(true_positions - predicted_positions, axis=1)
    
    # Compute metrics
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    percentile_90 = np.percentile(errors, 90)
    
    metrics = {
        'rmse': rmse,
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error,
        'percentile_90': percentile_90,
        'errors': errors
    }
    
    return metrics

def plot_results(true_positions, predicted_positions, metrics, output_path='localization_results.png'):
    errors = metrics['errors']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. CDF of errors
    ax = axes[0, 0]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cdf, linewidth=2)
    ax.axvline(metrics['median_error'], color='r', linestyle='--', label=f'Median: {metrics["median_error"]:.3f}m')
    ax.axvline(metrics['percentile_90'], color='g', linestyle='--', label=f'90%: {metrics["percentile_90"]:.3f}m')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Cumulative Distribution of Errors', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 2. Error histogram
    ax = axes[0, 1]
    ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_error'], color='r', linestyle='--', linewidth=2, label=f'Mean: {metrics["mean_error"]:.3f}m')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 3. 3D scatter plot (top view, color-coded by error)
    ax = axes[1, 0]
    scatter = ax.scatter(true_positions[:, 0], true_positions[:, 1], 
                        c=errors, cmap='hot_r', s=100, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top View: True Positions (colored by error)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Error (m)')
    
    # 4. Error vs position component
    ax = axes[1, 1]
    error_x = np.abs(true_positions[:, 0] - predicted_positions[:, 0])
    error_y = np.abs(true_positions[:, 1] - predicted_positions[:, 1])
    error_z = np.abs(true_positions[:, 2] - predicted_positions[:, 2])
    
    ax.boxplot([error_x, error_y, error_z], labels=['X', 'Y', 'Z'])
    ax.set_ylabel('Absolute Error (m)', fontsize=12)
    ax.set_title('Error by Coordinate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Results plot saved to: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("RF FINGERPRINTING LOCALIZATION")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset('../rf_dataset.pkl')
    
    np.random.seed(42)
    train_pos, train_feat, test_pos, test_feat = prepare_data(dataset, test_split=0.2)
    
    for k in [3, 5, 10]:
        print(f"\nPerforming kNN localization (k={k})...")
        predicted_pos = knn_localization(train_pos, train_feat, test_feat, k=k)
        print("✓ Localization complete")
        
        # Evaluation
        print(f"\nEvaluating performance for k={k}...")
        metrics = evaluate_performance(test_pos, predicted_pos)
        
        print("\n" + "=" * 60)
        print(f"PERFORMANCE METRICS (k={k})")
        print("=" * 60)
        print(f"  RMSE:           {metrics['rmse']:.4f} m")
        print(f"  Mean Error:     {metrics['mean_error']:.4f} m")
        print(f"  Median Error:   {metrics['median_error']:.4f} m")
        print(f"  90th %ile:      {metrics['percentile_90']:.4f} m")
        print(f"  Max Error:      {metrics['max_error']:.4f} m")
        print("=" * 60)
        
        # Visualize results
        print(f"\nGenerating visualization for k={k}...")
        np.savetxt(f"knn_errors_k{k}.csv", metrics["errors"], delimiter=",")
        plot_results(test_pos, predicted_pos, metrics, output_path=f'localization_results_k{k}.png')
    
    print("\n✅ Localization evaluation complete!")

if __name__ == "__main__":
    main()
