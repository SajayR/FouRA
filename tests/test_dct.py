import torch
import pytest
from src.foura.dct import DCT1D

@pytest.mark.parametrize("N", [8, 16, 32, 64]) # Test with different embedding sizes
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 50])
def test_dct1d_inverse(N, batch_size, seq_len):
    """Test if DCT1D inverse returns the original tensor."""
    dct_layer = DCT1D(N)
    # Create a random tensor
    x = torch.randn(batch_size, seq_len, N)

    # Apply DCT forward and inverse
    x_transformed = dct_layer.forward(x)
    x_reconstructed = dct_layer.inverse(x_transformed)

    # Check if the reconstructed tensor is close to the original
    assert torch.allclose(x_reconstructed, x, atol=1e-4), \
        f"DCT1D reconstruction failed for N={N}, batch={batch_size}, seq_len={seq_len}"

@pytest.mark.parametrize("N", [8, 16])
def test_dct1d_basis_orthonormality(N):
    """Test if the DCT basis matrix B is orthonormal (B @ B.T == I)."""
    dct_layer = DCT1D(N)
    # The basis matrix used for forward transform is BT.t() (B)
    # The basis matrix used for inverse transform is BT (B.T)
    # So we check BT.t() @ BT == I
    identity_matrix = torch.eye(N)
    basis_matrix = dct_layer.BT.t()
    product = basis_matrix @ dct_layer.BT
    assert torch.allclose(product, identity_matrix, atol=1e-5), \
        f"DCT1D basis is not orthonormal for N={N}"

    # Also check BT @ BT.t() == I
    product_inv = dct_layer.BT @ basis_matrix
    assert torch.allclose(product_inv, identity_matrix, atol=1e-5), \
        f"DCT1D inverse basis is not orthonormal for N={N}" 