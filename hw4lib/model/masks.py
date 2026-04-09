import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask for padding positions. 
    Args:
        padded_input: The input tensor, shape (N, T, ...).
        input_lengths: Actual lengths before padding, shape (N,).
    Returns:
        Boolean mask tensor with shape (N, T).
    """
    # Detect if we were passed (targets, lengths) or (lengths, max_len)
    if torch.is_tensor(padded_input) and padded_input.dim() > 1:
        # Standard case: (N, T, ...) and (N,)
        N = padded_input.size(0)
        T = padded_input.size(1)
        lengths = input_lengths
    else:
        # transformers.py call case: (lengths, max_len)
        lengths = padded_input
        N = lengths.size(0)
        T = input_lengths
        
    # Create mask: [N, T]
    # Indices [0, 1, ..., T-1]
    indices = torch.arange(T, device=lengths.device).expand(N, T)
    
    # Mask is True where index >= length
    mask = indices >= lengths.unsqueeze(1)
    
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a causal mask for self-attention. 
    Args:
        padded_input: Input tensor, shape (N, T, ...) or integer T.
    Returns:
        Boolean mask tensor with shape (T, T).
    """
    # Handle both tensor input (N, T, ...) and integer input (T)
    if torch.is_tensor(padded_input):
        T = padded_input.size(1)
        device = padded_input.device
    else:
        T = padded_input
        device = 'cpu' # Will be moved to device by .to() in transformers.py
        
    # Create upper triangular mask (True for values we want to mask)
    # diagonal=1 means we keep the diagonal as False (not masked)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    
    return mask

