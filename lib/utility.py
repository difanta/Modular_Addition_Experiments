import torch

def get_best_accelerator():
  device = None
  if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
  if device == None:
    device = torch.device("cpu")
  return device

def get_dft_basis(N, device='cpu', dtype=torch.complex64):
    """
    Returns the N x N unitary DFT matrix.
    """
    # Create indices from 0 to N-1
    n = torch.arange(N, device=device) 
    
    # Create a 2D grid of indices (N, N) using outer product
    j, k = torch.meshgrid(n, n, indexing='ij')
    
    # Compute the complex exponential
    # Note: In PyTorch, '1j' is the imaginary unit
    W = torch.exp(-2j * torch.pi * j * k / N)
    
    # Normalize to make it unitary (so that W @ W.conj().T == I)
    W = W / torch.sqrt(torch.tensor(float(N), device=device))

    return W.to(dtype)

def get_dht_basis(N, device='cpu', dtype=torch.float32):
    """
    Returns the N x N orthogonal DHT (Discrete Hartley Transform) matrix.
    
    The DHT is a real-valued transform closely related to the DFT.
    The kernel function is cas(x) = cos(x) + sin(x), and the resulting
    matrix is symmetric (H = H.T) and orthogonal (H @ H.T = I).
    """
    # Create indices from 0 to N-1
    n = torch.arange(N, device=device)

    # Create a 2D grid of indices (N, N) using outer product
    j, k = torch.meshgrid(n, n, indexing='ij')

    # Compute the cas function: cas(θ) = cos(θ) + sin(θ)
    # cas(2π·j·k/N)
    W = torch.cos(2 * torch.pi * j * k / N) + torch.sin(2 * torch.pi * j * k / N)

    # Normalize to make it orthogonal (so that W @ W.T == I)
    W = W / torch.sqrt(torch.tensor(float(N), device=device))

    return W.to(dtype)

def complex_sign(z):
    """Element-wise sign function for complex tensors (phase extraction)."""
    abs_z = torch.abs(z)
    # Avoid division by zero
    return torch.where(abs_z > 0, z / abs_z, torch.zeros_like(z))