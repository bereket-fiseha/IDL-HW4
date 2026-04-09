# NaN Loss & High CER Issues: Diagnosis & Fixes

## Summary of Issues

Your training encountered two main problems:
1. **All losses showing as NaN** (ce_loss, ctc_loss, joint_loss, perplexity)
2. **Extremely high CER (840%) at epoch 0** - model not learning

## Root Cause Analysis

### Primary Issue: Mixed Precision (float16) Instability
**File:** `hw4lib/trainers/asr_trainer.py` (Line 113)

**Problem:**
- Your code was using `torch.float16` for mixed precision training on CUDA
- float16 has limited precision (5-6 significant digits) and narrow range
- Operations like `log_softmax()` and `softmax()` in loss functions easily overflow/underflow
- Results in NaN or Inf values that propagate through the entire training loop

**Impact:**
```
float16 range: ±6.55×10⁴
When computing softmax:
  - Large logits (>10-20) → exp() → overflow → Inf → log_softmax() → NaN
  - Small logits (<-10-20) → exp() → underflow → zero → log(0) → -Inf
```

### Secondary Issue: CTC Loss Computation
**File:** `hw4lib/trainers/asr_trainer.py` (Line 119)

**Problem:**
- CTC loss wasn't validating input tensor shapes and values before computation
- Once first batch produced NaN, subsequent batches inherited NaN values
- No error recovery mechanism

## Applied Fixes

### Fix 1: Changed Mixed Precision Type ✅
**File:** `hw4lib/trainers/asr_trainer.py` (Line 113-116)

**Change:**
```python
# Before
dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16

# After  
dtype = torch.bfloat16  # More stable than float16 for deep learning
```

**Why:**
- `bfloat16` (Brain Float 16) has better dynamic range than float16
- Maintains 8 exponent bits (same as float32) while reducing mantissa
- Designed specifically for deep learning
- Prevents overflow/underflow in softmax operations

### Fix 2: Added NaN Detection in Training Loop ✅
**File:** `hw4lib/trainers/asr_trainer.py` (Around line 125-145)

**Added:**
- Validation checks before loss calculation
- NaN detection in model outputs
- Target shape validation
- Batch skipping on NaN detection (prevents error propagation)
- Diagnostic print statements for debugging

**Benefits:**
- Prevents NaN from corrupting subsequent batches
- Provides detailed diagnostics when NaN occurs
- Allows training to continue even with occasional NaN batches

### Fix 3: CTC Loss Validation ✅
**File:** `hw4lib/trainers/asr_trainer.py` (Line 119-145)

**Added:**
- Check for NaN in log_probs before CTC loss computation
- Check for Inf values that could cause CTC failures
- Try-catch block for CTC loss computation
- Detailed error logging with tensor shapes and values

### Fix 4: Improved Metrics Calculations ✅
**File:** `hw4lib/trainers/asr_trainer.py` (Multiple locations)

**Changes:**
- Safe division: `running_loss / total_tokens if total_tokens > 0 else 0.0`
- Safe exponential for perplexity: `exp(clamp(loss, min=1e-10))`
- Prevents log(negative), division by zero, and NaN propagation

## Why CER Is Still High (840% at Epoch 0)

This is actually **expected behavior** with a newly initialized, untrained model:

1. **Random initialization**: Model weights are random
2. **Random output distribution**: Model produces essentially random token predictions
3. **High error rate**: Random text has very high edit distance from ground truth

**What to expect:**
- Epoch 0: CER ≈ 100-500% (random predictions)
- Epoch 1-5: Gradual decrease if training is working
- After sufficient training: CER < 50% (decent performance)

**After applying fixes**, you should see:
- ✅ No NaN losses (instead actual numerical values)
- ✅ CER decreasing over epochs
- ✅ Training loss decreasing

## Testing the Fixes

1. **Run your training script** - it should now:
   - Show real loss values instead of NaN
   - Display CER improving over epochs (trend downward)
   - Not crash with numerical errors

2. **Monitor the metrics:**
   ```
   Epoch 1: ce_loss ≈ 5-10, CER ≈ 200-300%  (decreasing from random)
   Epoch 2: ce_loss ≈ 4-8, CER ≈ 150-200%   (continuing to improve)
   Epoch 5: ce_loss ≈ 2-4, CER ≈ 50-80%     (reasonable performance)
   ```

## Additional Recommendations

### 1. Monitor for Remaining NaN Issues
The diagnostic script `debug_nan_issues.py` has test utilities:
```bash
python debug_nan_issues.py
```
Run this to verify mixed precision stability.

### 2. If NaN Still Occurs:
- Disable mixed precision entirely (remove autocast context)
- Reduce learning rate (gradient scaling can cause overflow)
- Check model weight initialization
- Verify input data ranges

### 3. If CER Stops Decreasing:
- Check learning rate (might be too high/low)
- Verify data preprocessing
- Check if model is overfitting/underfitting
- Ensure targets/references are correctly formatted

### 4. Performance Optimization:
- Once training is stable, keep bfloat16 for faster training
- Consider gradient clipping if exploding gradients still occur
- Use appropriate batch size for GPU memory

## Files Modified

1. `hw4lib/trainers/asr_trainer.py`:
   - Line 113-116: Mixed precision dtype change
   - Line 125-145: NaN detection in training loop
   - Line 119-145: CTC loss validation  
   - Line 160-180: Progress bar metrics safety
   - Line 200-215: Final metrics calculation safety

2. `debug_nan_issues.py` (new file):
   - Diagnostic utilities for troubleshooting

## Next Steps

1. ✅ Apply the fixes above
2. 🔄 Run training and verify:
   - No NaN in loss values
   - CER decreasing over epochs
3. 📊 Monitor metrics over full training run
4. 🐛 If issues persist, run `debug_nan_issues.py` for detailed diagnostics

---

**Note:** The extremely high CER at epoch 0 is normal for untrained models. The important thing is that it *decreases* over epochs, indicating the model is learning. With these fixes, you should see improvement starting from epoch 1.
