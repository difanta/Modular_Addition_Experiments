import torch

def get_dft_basis(N, device='cpu', dtype=torch.complex128):
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

def complex_sign(z):
    """Element-wise sign function for complex tensors (phase extraction)."""
    abs_z = torch.abs(z)
    # Avoid division by zero
    return torch.where(abs_z > 0, z / abs_z, torch.zeros_like(z))

def calculate_sparse_code(activations, basis, lambd):
    sparse_code = basis.mH @ activations @ basis

    return complex_sign(sparse_code)*torch.clamp(torch.abs(sparse_code) - (lambd/2.0), min=0.0)

def basis_update(activations, basis, sparse_code, max_line_search=20):
    # compute euclidean gradient

    eucl_grad = 2.0 * torch.sum(activations.mH @ basis @ sparse_code + activations @ basis @ sparse_code.mH, dim=0)

    M = basis.mH @ eucl_grad
    rienmann_grad = eucl_grad - basis @ (0.5 * (M + M.mH))

    grad_norm = torch.linalg.norm(rienmann_grad)
    if grad_norm > 0:
        rienmann_grad = rienmann_grad / (grad_norm + 1e-8)

    # 3. Backtracking line search for step size alpha
    alpha = 1.0
    # We evaluate loss ignoring L1 because sparse_code is fixed during this step
    current_loss = compute_loss(activations, basis, sparse_code, 0.0)
    
    for _ in range(max_line_search):
        # Retract onto the unitary manifold using QR decomposition
        # torch.linalg.qr returns Q, R
        Q, R = torch.linalg.qr(basis + alpha * rienmann_grad)
        
        # Optional: Normalize phases of Q for numerical stability
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        phases = torch.exp(-1j * torch.angle(diag_R))
        Q = Q * phases.unsqueeze(0)
        
        new_loss = compute_loss(activations, Q, sparse_code, 0.0)
        
        # We are doing gradient ascent on the trace, which minimizes the reconstruction error
        if new_loss < current_loss * (1.0 - 1e-12):
            return Q
            
        alpha *= 0.5

    # fallback if line search reaches maximum iterations
    return basis

def compute_loss(activations, basis, sparse_code, lambd):
    """Compute the objective function value."""

    return (torch.linalg.matrix_norm(activations - basis @ sparse_code @ basis.mH, ord='fro') ** 2).sum() + lambd * torch.abs(sparse_code).sum()


def find_basis(activations, tolerance=1e-1, max_steps=3000, lambd = 1e-1):
    """
    Find a shared sparse basis L for a set of activation matrices.
    Activations shape expected: (L, N, N, D) or (N, N, D)
    """
    activations = activations.to(torch.complex64)
        
    if activations.dim() == 3:
        activations = activations.unsqueeze(0)
        
    if activations.dim() == 4:
        L_layers, N, _, D = activations.shape
        activations = activations.permute(0, 3, 1, 2).reshape(L_layers * D, N, N)
    else:
        raise ValueError(f"Expected activations of shape (L, N, N, D) or (N, N, D), but got {activations.shape}")

    basis, _ = torch.linalg.qr(torch.randn(activations.size(1), activations.size(1), dtype=torch.complex64, device=activations.device))

    #basis = torch.eye(activations.size(1), dtype=torch.complex64, device=activations.device)
    
    #basis = get_dft_basis(activations.size(1), dtype=torch.complex64, device=activations.device)

    error = []
    for idx in range(max_steps):
        try:
            sparse_code = calculate_sparse_code(activations, basis, lambd)
            basis = basis_update(activations, basis, sparse_code)

            error.append(compute_loss(activations, basis, sparse_code, lambd).item())

            if error[-1] < tolerance:
                print(f"Reached desired tolerance in {idx} steps")
                break
            else:
                print(f"Iteration: {idx}, error: {error[-1]}")
        except KeyboardInterrupt:
            return error, basis

    print(f"Final Loss: {error[-1]:.6f}")
    return error, basis