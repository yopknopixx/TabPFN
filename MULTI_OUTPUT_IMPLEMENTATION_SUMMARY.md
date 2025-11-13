# TabPFN Multi-Output Regression Implementation Summary

## Overview
This document summarizes the implementation of multi-output regression support for TabPFN using **Option 2: Multiple Independent Decoder Heads**.

## Architecture Changes

### 1. Transformer Architecture (`src/tabpfn/architectures/base/transformer.py`)

**Changes:**
- Added `n_regression_outputs` parameter to `PerFeatureTransformer.__init__()`
- Modified decoder construction to create separate decoder heads when `n_regression_outputs > 1`
- Each decoder head is an independent `nn.Sequential` with shape: `Linear(ninp, nhid) → GELU → Linear(nhid, n_out)`
- Updated `forward()` method to stack outputs from all decoder heads
- Multi-output shape: `(seq_len, n_outputs, n_out)` where `n_out` is `num_bars`

**Key Code Sections:**
- Lines 111, 250-280: Decoder initialization with multiple heads
- Lines 629-638: Forward pass aggregation of multi-output logits

### 2. Regressor (`src/tabpfn/regressor.py`)

**Major Changes:**

#### Constructor:
- Added `n_outputs` parameter (default=1) to specify number of output targets
- Stored as `self.n_outputs` for use during fitting

#### Fitting:
- **Multi-output target handling** (lines 656-661):
  - Automatically detects number of outputs from `y.shape`
  - Reshapes 1D `y` to 2D for consistency
  - Sets `self.n_outputs_` based on actual data

- **Per-output normalization** (lines 860-889):
  - Single-output: Uses scalar mean/std (backward compatible)
  - Multi-output: Uses per-output mean/std arrays with shape `(n_outputs,)`
  - Normalizes each output independently: `(y - mean) / std`

- **Per-output bar distributions** (lines 815-889):
  - Single-output: Single `FullSupportBarDistribution` instance
  - Multi-output: List of `FullSupportBarDistribution`, one per output
  - Stored in `self.znorm_space_bardist_` and `self.raw_space_bardist_`

#### Forward Pass (lines 1136-1223):
- Handles borders for single vs multi-output
- For multi-output: Processes each output's borders separately
- Applies logit masks per-output when needed
- Output shape: `(n_samples, n_outputs, num_bars)` for multi-output

#### Prediction (lines 1016-1079):
- **Single-output path**: Original behavior preserved
- **Multi-output path**:
  - Iterates over each output dimension
  - Extracts logits for specific output: `logits[:, output_idx, :]`
  - Translates borders independently
  - Stacks all outputs: `(n_samples, n_outputs, num_bars)`

#### Helper Functions:
- **`_logits_to_output()`** (lines 1350-1410):
  - Handles both single and multi-output cases
  - Multi-output returns shape `(n_samples, n_outputs)` for mean/median/mode
  - Quantiles return list of arrays, each with shape `(n_samples, n_outputs)`

- **`_handle_constant_target()`** (lines 1286-1333):
  - Single-output: Returns `(n_samples,)` array
  - Multi-output: Returns `(n_samples, n_outputs)` array

### 3. Architecture Factory (`src/tabpfn/architectures/base/__init__.py`)

**Changes:**
- Added `n_regression_outputs` parameter to `get_architecture()` function (line 62)
- Passes parameter to `PerFeatureTransformer` constructor (line 110)
- Defaults to 1 for backward compatibility

## Usage

### Single-Output (Backward Compatible)
```python
from tabpfn import TabPFNRegressor

model = TabPFNRegressor(n_estimators=8)
model.fit(X_train, y_train)  # y_train shape: (n_samples,) or (n_samples, 1)
predictions = model.predict(X_test)  # shape: (n_samples,)
```

### Multi-Output
```python
from tabpfn import TabPFNRegressor

# Specify number of outputs upfront
model = TabPFNRegressor(n_outputs=3, n_estimators=8)
model.fit(X_train, y_train)  # y_train shape: (n_samples, 3)
predictions = model.predict(X_test)  # shape: (n_samples, 3)

# Different output types
predictions_median = model.predict(X_test, output_type="median")  # (n_samples, 3)
predictions_mode = model.predict(X_test, output_type="mode")  # (n_samples, 3)
predictions_quantiles = model.predict(X_test, output_type="quantiles")  # list of arrays
```

## Key Features

### ✅ Implemented
1. **Multiple independent decoder heads** - One per output
2. **Per-output normalization** - Independent mean/std for each output
3. **Per-output bar distributions** - Separate probability distributions
4. **Backward compatibility** - Single-output still works exactly as before
5. **All output types supported** - mean, median, mode, quantiles, main, full
6. **Constant target handling** - Works for both single and multi-output

### ⚠️ Limitations & Notes

1. **Pretrained Model Weights**:
   - Current pretrained models are trained with single output head
   - When `n_outputs > 1`, new decoder heads are created with **random initialization**
   - For best results with multi-output, consider **fine-tuning** after initialization

2. **Y-Encoder**:
   - Current implementation processes targets with existing single-target y_encoder
   - May need enhancement for truly multi-dimensional target encoding in future

3. **Model Loading**:
   - When loading pretrained models, weights are loaded for shared components (encoder, transformer layers)
   - Additional decoder heads (when `n_outputs > 1`) are initialized randomly

## Testing

A comprehensive test script is available at `test_multioutput.py` that validates:
- Single-output regression (backward compatibility)
- Multi-output regression with 2 and 3 targets
- Different prediction output types
- Shape validation for all outputs

## Files Modified

1. `src/tabpfn/architectures/base/transformer.py` - Core transformer with multiple heads
2. `src/tabpfn/regressor.py` - Multi-output regression logic
3. `src/tabpfn/architectures/base/__init__.py` - Architecture factory updates
4. `test_multioutput.py` - Comprehensive test suite

## Next Steps

To fully utilize multi-output with pretrained models:

1. **Fine-tuning**: After initializing with `n_outputs > 1`, fine-tune the model on your multi-output dataset
2. **Training from scratch**: For best multi-output performance, train TabPFN from scratch with multi-output data
3. **Y-encoder enhancement**: Consider extending y_encoder to better handle multi-dimensional targets

## Performance Considerations

- **Memory**: Multi-output uses more memory due to multiple decoder heads
- **Speed**: Slightly slower due to processing multiple outputs, but benefits from shared encoder
- **Quality**: Initial predictions may be less accurate until fine-tuned, as new heads are randomly initialized

## Conclusion

The implementation provides a flexible multi-output regression capability while maintaining full backward compatibility with single-output use cases. The architecture uses independent decoder heads, allowing each output to learn its own transformation from the shared latent representation.
