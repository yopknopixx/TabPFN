"""
Benchmark script for comparing multi-output TabPFN vs multiple single-output TabPFN models.
Tests on the Sarcos Robot Arm dataset (7 joint torques).
"""

import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
from scipy.io import loadmat
import os

# Import TabPFN
from tabpfn import TabPFNRegressor


def download_sarcos_dataset(cache_dir='./data'):
    """
    Download the Sarcos dataset.
    Returns X_train, y_train, X_test, y_test

    Dataset info:
    - Training samples: 44,484
    - Test samples: 4,449
    - Features: 21 (joint positions, velocities, accelerations)
    - Targets: 7 (joint torques)
    """
    os.makedirs(cache_dir, exist_ok=True)

    train_file = os.path.join(cache_dir, 'sarcos_inv.mat')
    test_file = os.path.join(cache_dir, 'sarcos_inv_test.mat')

    # URLs for Sarcos dataset
    train_url = 'http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat'
    test_url = 'http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat'

    # Download training data
    if not os.path.exists(train_file):
        print("Downloading Sarcos training data...")
        response = requests.get(train_url, timeout=60)
        with open(train_file, 'wb') as f:
            f.write(response.content)
        print("Training data downloaded.")
    else:
        print("Using cached training data.")

    # Download test data
    if not os.path.exists(test_file):
        print("Downloading Sarcos test data...")
        response = requests.get(test_url, timeout=60)
        with open(test_file, 'wb') as f:
            f.write(response.content)
        print("Test data downloaded.")
    else:
        print("Using cached test data.")

    # Load data
    train_data = loadmat(train_file)['sarcos_inv']
    test_data = loadmat(test_file)['sarcos_inv_test']

    # Split features and targets
    X_train = train_data[:, :21]
    y_train = train_data[:, 21:]
    X_test = test_data[:, :21]
    y_test = test_data[:, 21:]

    print(f"\nDataset loaded:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Targets: {y_train.shape[1]}")

    return X_train, y_train, X_test, y_test


def subsample_dataset(X_train, y_train, X_test, y_test, train_size=1000, test_size=200):
    """
    Subsample the dataset to a smaller size for faster testing.
    TabPFN works best with smaller datasets anyway.
    """
    if X_train.shape[0] > train_size:
        indices = np.random.choice(X_train.shape[0], train_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    if X_test.shape[0] > test_size:
        indices = np.random.choice(X_test.shape[0], test_size, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]

    print(f"\nSubsampled dataset:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")

    return X_train, y_train, X_test, y_test


def evaluate_predictions(y_true, y_pred, target_names=None):
    """
    Calculate MSE, MAE, and R² for each target and overall.
    """
    n_targets = y_true.shape[1]

    if target_names is None:
        target_names = [f"Joint {i+1}" for i in range(n_targets)]

    results = {
        'per_target': [],
        'overall': {}
    }

    # Per-target metrics
    for i in range(n_targets):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        results['per_target'].append({
            'target': target_names[i],
            'mse': mse,
            'mae': mae,
            'r2': r2
        })

    # Overall metrics (averaged across targets)
    overall_mse = np.mean([r['mse'] for r in results['per_target']])
    overall_mae = np.mean([r['mae'] for r in results['per_target']])
    overall_r2 = np.mean([r['r2'] for r in results['per_target']])

    results['overall'] = {
        'mse': overall_mse,
        'mae': overall_mae,
        'r2': overall_r2
    }

    return results


def benchmark_multi_output_model(X_train, y_train, X_test, y_test, n_estimators=8):
    """
    Benchmark using a single multi-output TabPFN model.
    """
    print("\n" + "="*60)
    print("APPROACH 1: One Multi-Output Model")
    print("="*60)

    n_outputs = y_train.shape[1]

    # Training
    print(f"\nTraining TabPFNRegressor with n_outputs={n_outputs}...")
    model = TabPFNRegressor(n_outputs=n_outputs, n_estimators=n_estimators)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    # Prediction
    print(f"Making predictions on {X_test.shape[0]} test samples...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    print(f"Prediction completed in {pred_time:.2f} seconds")

    # Evaluation
    results = evaluate_predictions(y_test, y_pred)

    return {
        'model': model,
        'predictions': y_pred,
        'results': results,
        'train_time': train_time,
        'pred_time': pred_time
    }


def benchmark_single_output_models(X_train, y_train, X_test, y_test, n_estimators=8):
    """
    Benchmark using separate single-output TabPFN models (one per target).
    """
    print("\n" + "="*60)
    print("APPROACH 2: Multiple Single-Output Models")
    print("="*60)

    n_outputs = y_train.shape[1]
    models = []
    predictions = []

    total_train_time = 0
    total_pred_time = 0

    for i in range(n_outputs):
        print(f"\n--- Target {i+1}/{n_outputs}: Joint {i+1} ---")

        # Training
        model = TabPFNRegressor(n_estimators=n_estimators)

        start_time = time.time()
        model.fit(X_train, y_train[:, i])
        train_time = time.time() - start_time
        total_train_time += train_time

        print(f"Training completed in {train_time:.2f} seconds")

        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        total_pred_time += pred_time

        print(f"Prediction completed in {pred_time:.2f} seconds")

        models.append(model)
        predictions.append(y_pred)

    # Combine predictions
    y_pred_combined = np.column_stack(predictions)

    print(f"\nTotal training time: {total_train_time:.2f} seconds")
    print(f"Total prediction time: {total_pred_time:.2f} seconds")

    # Evaluation
    results = evaluate_predictions(y_test, y_pred_combined)

    return {
        'models': models,
        'predictions': y_pred_combined,
        'results': results,
        'train_time': total_train_time,
        'pred_time': total_pred_time
    }


def print_results_comparison(multi_output_results, single_output_results):
    """
    Print a detailed comparison of both approaches.
    """
    print("\n" + "="*80)
    print(" RESULTS COMPARISON")
    print("="*80)

    # Performance metrics
    print("\n📊 PERFORMANCE METRICS (on test set)")
    print("-" * 80)

    mo_results = multi_output_results['results']
    so_results = single_output_results['results']

    # Per-target comparison
    print("\nPer-Target Metrics:")
    print(f"{'Target':<15} {'Metric':<8} {'Multi-Output':<20} {'Single-Output':<20} {'Winner':<10}")
    print("-" * 80)

    for i, (mo_target, so_target) in enumerate(zip(mo_results['per_target'], so_results['per_target'])):
        target_name = mo_target['target']

        # MSE
        mo_mse = mo_target['mse']
        so_mse = so_target['mse']
        winner_mse = "Multi-Out" if mo_mse < so_mse else "Single-Out"
        print(f"{target_name:<15} {'MSE':<8} {mo_mse:<20.6f} {so_mse:<20.6f} {winner_mse:<10}")

        # MAE
        mo_mae = mo_target['mae']
        so_mae = so_target['mae']
        winner_mae = "Multi-Out" if mo_mae < so_mae else "Single-Out"
        print(f"{'':<15} {'MAE':<8} {mo_mae:<20.6f} {so_mae:<20.6f} {winner_mae:<10}")

        # R²
        mo_r2 = mo_target['r2']
        so_r2 = so_target['r2']
        winner_r2 = "Multi-Out" if mo_r2 > so_r2 else "Single-Out"
        print(f"{'':<15} {'R²':<8} {mo_r2:<20.6f} {so_r2:<20.6f} {winner_r2:<10}")
        print()

    # Overall comparison
    print("\nOverall Metrics (averaged across all targets):")
    print(f"{'Metric':<15} {'Multi-Output':<20} {'Single-Output':<20} {'Improvement':<15}")
    print("-" * 80)

    mo_overall = mo_results['overall']
    so_overall = so_results['overall']

    # MSE
    mse_improvement = ((so_overall['mse'] - mo_overall['mse']) / so_overall['mse'] * 100)
    print(f"{'MSE':<15} {mo_overall['mse']:<20.6f} {so_overall['mse']:<20.6f} {mse_improvement:>+.2f}%")

    # MAE
    mae_improvement = ((so_overall['mae'] - mo_overall['mae']) / so_overall['mae'] * 100)
    print(f"{'MAE':<15} {mo_overall['mae']:<20.6f} {so_overall['mae']:<20.6f} {mae_improvement:>+.2f}%")

    # R²
    r2_improvement = ((mo_overall['r2'] - so_overall['r2']) / abs(so_overall['r2']) * 100)
    print(f"{'R²':<15} {mo_overall['r2']:<20.6f} {so_overall['r2']:<20.6f} {r2_improvement:>+.2f}%")

    # Timing comparison
    print("\n⏱️  TIMING COMPARISON")
    print("-" * 80)

    print(f"{'Metric':<20} {'Multi-Output':<20} {'Single-Output':<20} {'Speedup':<15}")
    print("-" * 80)

    mo_train = multi_output_results['train_time']
    so_train = single_output_results['train_time']
    train_speedup = so_train / mo_train
    print(f"{'Training Time':<20} {mo_train:<20.2f}s {so_train:<20.2f}s {train_speedup:.2f}x")

    mo_pred = multi_output_results['pred_time']
    so_pred = single_output_results['pred_time']
    pred_speedup = so_pred / mo_pred
    print(f"{'Prediction Time':<20} {mo_pred:<20.2f}s {so_pred:<20.2f}s {pred_speedup:.2f}x")

    total_mo = mo_train + mo_pred
    total_so = so_train + so_pred
    total_speedup = total_so / total_mo
    print(f"{'Total Time':<20} {total_mo:<20.2f}s {total_so:<20.2f}s {total_speedup:.2f}x")

    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)

    if mo_overall['mse'] < so_overall['mse']:
        perf_winner = "Multi-Output model"
        perf_margin = mse_improvement
    else:
        perf_winner = "Single-Output models"
        perf_margin = -mse_improvement

    print(f"\n✓ Performance: {perf_winner} wins by {abs(perf_margin):.2f}% (MSE)")
    print(f"✓ Speed: Multi-Output is {train_speedup:.2f}x faster for training, {pred_speedup:.2f}x faster for prediction")
    print(f"✓ Model Count: 1 model vs {len(single_output_results['models'])} models")

    print("\n" + "="*80)


def main():
    """
    Main benchmark function.
    """
    print("="*80)
    print(" TABPFN MULTI-OUTPUT BENCHMARK: SARCOS ROBOT ARM DATASET")
    print("="*80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Download and load dataset
    X_train, y_train, X_test, y_test = download_sarcos_dataset()

    # Subsample to TabPFN-friendly size
    # TabPFN works best with datasets of up to ~10k samples
    X_train, y_train, X_test, y_test = subsample_dataset(
        X_train, y_train, X_test, y_test,
        train_size=1000,  # Adjust as needed
        test_size=200
    )

    # Run benchmarks
    n_estimators = 8  # Number of ensemble models

    print(f"\nUsing n_estimators={n_estimators} for both approaches")

    # Approach 1: Multi-output model
    multi_output_results = benchmark_multi_output_model(
        X_train, y_train, X_test, y_test, n_estimators=n_estimators
    )

    # Approach 2: Single-output models
    single_output_results = benchmark_single_output_models(
        X_train, y_train, X_test, y_test, n_estimators=n_estimators
    )

    # Print comparison
    print_results_comparison(multi_output_results, single_output_results)

    return multi_output_results, single_output_results


if __name__ == "__main__":
    results = main()
