"""
Diagnostic script to identify where NaN values are coming from in training.
Run this to get detailed information about where NaN appears in the training loop.
"""

import torch
import torch.nn as nn
import sys

def check_for_nan(tensor, name="tensor"):
    """Check if tensor contains NaN and print diagnostic info"""
    if isinstance(tensor, torch.Tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            print(f"\n⚠️  {name}:")
            print(f"   Shape: {tensor.shape}")
            print(f"   Dtype: {tensor.dtype}")
            print(f"   Has NaN: {has_nan}")
            print(f"   Has Inf: {has_inf}")
            print(f"   Min: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'N/A'}")
            print(f"   Max: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'N/A'}")
            print(f"   Mean: {tensor[~torch.isnan(tensor)].mean().item() if (~torch.isnan(tensor)).any() else 'N/A'}")
            return True
    return False

def debug_mixed_precision_issues():
    """Demonstrate float16 instability issues"""
    print("=" * 60)
    print("MIXED PRECISION STABILITY CHECK")
    print("=" * 60)
    
    # Test LogSoftmax in float16 vs float32
    print("\n1. LogSoftmax Stability Test:")
    x = torch.randn(4, 10)
    
    # Test with float32
    x_f32 = x.float()
    log_probs_f32 = torch.nn.functional.log_softmax(x_f32, dim=-1)
    print(f"   float32 - NaN count: {torch.isnan(log_probs_f32).sum().item()}")
    
    # Test with float16
    x_f16 = x.half()
    log_probs_f16 = torch.nn.functional.log_softmax(x_f16, dim=-1)
    print(f"   float16 - NaN count: {torch.isnan(log_probs_f16).sum().item()}")
    
    # Test with extreme values
    print("\n2. Extreme Values in float16:")
    x_extreme = torch.randn(4, 1000) * 1000  # Large values
    x_extreme_f16 = x_extreme.half()
    log_probs_extreme = torch.nn.functional.log_softmax(x_extreme_f16, dim=-1)
    print(f"   Large values - NaN count: {torch.isnan(log_probs_extreme).sum().item()}")
    print(f"   Large values - Inf count: {torch.isinf(log_probs_extreme).sum().item()}")

def debug_ctc_loss_issues():
    """Test CTCLoss with different input scenarios"""
    print("\n" + "=" * 60)
    print("CTC LOSS STABILITY CHECK")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 50
    vocab_size = 100
    target_len = 20
    
    # Create test inputs
    log_probs = torch.nn.functional.log_softmax(torch.randn(seq_len, batch_size, vocab_size), dim=-1)
    targets = torch.randint(1, vocab_size, (batch_size, target_len))
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    target_lengths = torch.randint(5, target_len+1, (batch_size,), dtype=torch.long)
    
    # Test CTC loss
    print("\n1. Basic CTC Loss Test:")
    for blank_id in [0, -1]:
        try:
            criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            has_nan = torch.isnan(loss).item() if isinstance(loss, torch.Tensor) else False
            print(f"   blank_id={blank_id}: Loss={loss.item():.4f}, NaN={has_nan}")
        except Exception as e:
            print(f"   blank_id={blank_id}: ERROR - {e}")
    
    # Test with float16
    print("\n2. CTC Loss with float16:")
    log_probs_f16 = log_probs.half()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    try:
        loss_f16 = criterion(log_probs_f16, targets, input_lengths, target_lengths)
        print(f"   Loss: {loss_f16.item():.4f}, NaN: {torch.isnan(loss_f16).item()}")
    except Exception as e:
        print(f"   ERROR: {e}")

def debug_cross_entropy_issues():
    """Test CrossEntropyLoss with different scenarios"""
    print("\n" + "=" * 60)
    print("CROSS ENTROPY LOSS STABILITY CHECK")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 20
    vocab_size = 100
    
    # Create test inputs
    logits = torch.randn(batch_size * seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    pad_id = 0
    
    # Test with float32
    print("\n1. CrossEntropyLoss with float32:")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss_f32 = criterion(logits.float(), targets)
    print(f"   Loss: {loss_f32.item():.4f}, NaN: {torch.isnan(loss_f32).item()}")
    
    # Test with float16
    print("\n2. CrossEntropyLoss with float16:")
    logits_f16 = logits.half()
    try:
        loss_f16 = criterion(logits_f16, targets)
        print(f"   Loss: {loss_f16.item():.4f}, NaN: {torch.isnan(loss_f16).item()}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test with extreme values
    print("\n3. CrossEntropyLoss with extreme values (float32):")
    logits_extreme = torch.randn(batch_size * seq_len, vocab_size) * 1000
    loss_extreme = criterion(logits_extreme, targets)
    print(f"   Loss: {loss_extreme.item():.4f}, NaN: {torch.isnan(loss_extreme).item()}")

def main():
    print("\n" + "=" * 60)
    print("NaN DEBUGGING SUITE")
    print("=" * 60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    debug_mixed_precision_issues()
    debug_ctc_loss_issues()
    debug_cross_entropy_issues()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("""
1. **Disable float16 or switch to bfloat16**:
   - float16 has poor numerical stability for softmax/logsoftmax
   - Either disable autocast entirely or use bfloat16
   
2. **File: hw4lib/trainers/asr_trainer.py, Line 113-114**:
   - Change: `dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16`
   - To: `dtype = torch.bfloat16`  # Use bfloat16 exclusively
   - Or remove autocast entirely for stability
   
3. **Verify CTC loss inputs**:
   - Ensure input_lengths, target_lengths, and targets have correct shapes
   - Ensure targets don't contain padding tokens or invalid values
   
4. **Check model initialization**:
   - Verify weights are in reasonable range (not NaN or Inf)
   - Check embedding layers are initialized properly
   """)

if __name__ == "__main__":
    main()
