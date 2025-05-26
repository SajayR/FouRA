import torch
import pytest
from src.foura.foura import FouRA

@pytest.mark.parametrize("N", [32, 64])
@pytest.mark.parametrize("rank", [4, 8, 16])
@pytest.mark.parametrize("transform_type", ["none", "fft", "dct"])
@pytest.mark.parametrize("use_gate", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [10, 20])
def test_foura_module_output_shape(N, rank, transform_type, use_gate, batch_size, seq_len):
    """Test if the FouRA module output has the correct shape."""
    if rank > N // 2 +1 and transform_type == "fft": # Rank for fft is N//2+1
        pytest.skip("Rank is too large for FFT freq_dim")
    if rank > N and (transform_type == "dct" or transform_type == "none") :
        pytest.skip("Rank is too large for DCT/None freq_dim")

    model = FouRA(N=N, rank=rank, transform_type=transform_type, use_gate=use_gate)
    x = torch.randn(batch_size, seq_len, N)
    
    # For FFT, input to FouRA is real, but internal ops can be complex.
    # Ensure model handles this if input is float.
    if transform_type == "fft" and x.dtype == torch.cfloat:
        # This case should not happen as rfft expects real input
        # but if it did, the linear layers might need to be torch.cdouble
        pass # Or raise an error if specific handling is needed
        
    output = model(x)
    assert output.shape == x.shape, \
        f"FouRA module output shape incorrect. Expected {x.shape}, got {output.shape}"
    assert output.dtype == x.dtype, \
        f"FouRA module output dtype incorrect. Expected {x.dtype}, got {output.dtype}"

@pytest.mark.parametrize("N, rank, transform_type, use_gate", [
    (32, 8, "dct", True),
    (64, 16, "fft", True),
    (32, 8, "none", True)
])
def test_foura_gating_parameters_created(N, rank, transform_type, use_gate):
    """Test if gating parameters (alpha, beta) are created when use_gate is True."""
    model = FouRA(N=N, rank=rank, transform_type=transform_type, use_gate=use_gate)
    if use_gate:
        assert hasattr(model, 'alpha'), "Model should have 'alpha' parameter for gating."
        assert hasattr(model, 'beta'), "Model should have 'beta' parameter for gating."
        assert model.alpha.shape == (rank,)
        assert model.beta.shape == (rank,)
    else:
        assert not hasattr(model, 'alpha'), "Model should NOT have 'alpha' if use_gate=False."
        assert not hasattr(model, 'beta'), "Model should NOT have 'beta' if use_gate=False."

@pytest.mark.parametrize("N, rank, transform_type", [
    (32, 8, "fft"),
])
def test_foura_fft_complex_handling(N, rank, transform_type):
    """Test FouRA with FFT specifically for complex dtype consistency."""
    model = FouRA(N=N, rank=rank, transform_type=transform_type, use_gate=False)
    x_real = torch.randn(2, 10, N) # Batch=2, SeqLen=10, Features=N
    output_real = model(x_real)
    assert output_real.is_complex() == False, "Output should be real if input is real for FFT"

    # Check if internal layers are complex as expected for FFT
    assert model.down.weight.is_complex(), "Down projection should be complex for FFT"
    assert model.up.weight.is_complex(), "Up projection should be complex for FFT" 