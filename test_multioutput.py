"""Test script for multi-output regression with TabPFN."""

import numpy as np
from sklearn.datasets import make_regression
from tabpfn import TabPFNRegressor

# Test 1: Single-output regression (backward compatibility)
print("=" * 60)
print("Test 1: Single-output regression (backward compatibility)")
print("=" * 60)

X_single, y_single = make_regression(
    n_samples=100,
    n_features=10,
    n_targets=1,
    noise=0.1,
    random_state=42
)

print(f"X shape: {X_single.shape}")
print(f"y shape: {y_single.shape}")

model_single = TabPFNRegressor(n_estimators=2, random_state=42)
print("\nFitting single-output model...")
model_single.fit(X_single, y_single)
print(f"Model n_outputs_: {model_single.n_outputs_}")

pred_single = model_single.predict(X_single[:10])
print(f"Predictions shape: {pred_single.shape}")
print(f"First 5 predictions: {pred_single[:5]}")

# Test 2: Multi-output regression
print("\n" + "=" * 60)
print("Test 2: Multi-output regression")
print("=" * 60)

X_multi, y_multi = make_regression(
    n_samples=100,
    n_features=10,
    n_targets=3,  # 3 outputs
    noise=0.1,
    random_state=42
)

print(f"X shape: {X_multi.shape}")
print(f"y shape: {y_multi.shape}")

model_multi = TabPFNRegressor(
    n_outputs=3,  # Specify number of outputs
    n_estimators=2,
    random_state=42
)
print("\nFitting multi-output model...")
try:
    model_multi.fit(X_multi, y_multi)
    print(f"Model n_outputs_: {model_multi.n_outputs_}")
    print(f"y_train_mean_ shape: {model_multi.y_train_mean_.shape if hasattr(model_multi, 'y_train_mean_') else 'N/A'}")
    print(f"y_train_std_ shape: {model_multi.y_train_std_.shape if hasattr(model_multi, 'y_train_std_') else 'N/A'}")

    pred_multi = model_multi.predict(X_multi[:10])
    print(f"\nPredictions shape: {pred_multi.shape}")
    print(f"First prediction (all outputs): {pred_multi[0]}")
    print(f"First 3 predictions:\n{pred_multi[:3]}")

    # Test different output types
    print("\n" + "-" * 60)
    print("Testing different output types:")
    print("-" * 60)

    pred_median = model_multi.predict(X_multi[:5], output_type="median")
    print(f"Median predictions shape: {pred_median.shape}")

    pred_mode = model_multi.predict(X_multi[:5], output_type="mode")
    print(f"Mode predictions shape: {pred_mode.shape}")

    pred_quantiles = model_multi.predict(X_multi[:5], output_type="quantiles")
    print(f"Quantiles type: {type(pred_quantiles)}")
    print(f"Number of quantiles: {len(pred_quantiles)}")
    print(f"First quantile shape: {pred_quantiles[0].shape}")

except Exception as e:
    print(f"Error during multi-output regression: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multi-output with 2 targets
print("\n" + "=" * 60)
print("Test 3: Multi-output regression with 2 targets")
print("=" * 60)

X_two, y_two = make_regression(
    n_samples=100,
    n_features=10,
    n_targets=2,
    noise=0.1,
    random_state=42
)

print(f"X shape: {X_two.shape}")
print(f"y shape: {y_two.shape}")

model_two = TabPFNRegressor(n_outputs=2, n_estimators=2, random_state=42)
print("\nFitting 2-output model...")
try:
    model_two.fit(X_two, y_two)
    pred_two = model_two.predict(X_two[:5])
    print(f"Predictions shape: {pred_two.shape}")
    print(f"Predictions:\n{pred_two}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
