import torch
import torch.nn as nn
import pytest
from src.foura.wrappers import FouRAConfig, FouRAInjectedLinear, get_foura_model, _matches, _get_parent
from src.foura.foura import FouRA

@pytest.fixture
def dummy_linear_model():
    """Provides a simple nn.Module with a Linear layer for testing injection."""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model.sub_module = nn.Sequential(nn.Linear(10,5))
    return model

@pytest.fixture
def simple_model_with_target_name():
    model = nn.Sequential(nn.Linear(10,10, bias=False))
    model.vit_block = nn.Sequential(nn.Linear(20,20, bias=False))
    model.vit_block.attention = nn.Sequential(nn.Linear(30,30, bias=False))
    return model

# Tests for FouRAConfig
def test_foura_config_defaults():
    cfg = FouRAConfig()
    assert cfg.rank == 16
    assert cfg.foura_alpha == 32.0
    assert cfg.transform_type == "dct"
    assert cfg.target_modules == ["query", "value"]
    assert cfg.use_gate == False

def test_foura_config_custom():
    cfg = FouRAConfig(rank=8, foura_alpha=16.0, transform_type="fft", target_modules=["attention"], use_gate=True)
    assert cfg.rank == 8
    assert cfg.foura_alpha == 16.0
    assert cfg.transform_type == "fft"
    assert cfg.target_modules == ["attention"]
    assert cfg.use_gate == True

def test_foura_config_post_init_target_modules_none():
    cfg = FouRAConfig(target_modules=None)
    assert cfg.target_modules == ["query", "value"] # Default from post_init

def test_foura_config_transform_type_case_insensitivity():
    cfg = FouRAConfig(transform_type="FFT ")
    assert cfg.transform_type == "fft"

# Tests for _matches and _get_parent (helper functions)
def test_matches_function():
    assert _matches("encoder.layer.0.attention.query", ["query", "value"]) == True
    assert _matches("encoder.layer.0.attention.key", ["query", "value"]) == False
    assert _matches("decoder.output_linear", ["output"]) == True
    assert _matches("some_other_layer", ["query"]) == False

def test_get_parent_function(dummy_linear_model):
    model = dummy_linear_model
    parent, name = _get_parent(model, "0")
    assert parent == model
    assert name == "0"
    assert isinstance(getattr(parent,name), nn.Linear)

    parent, name = _get_parent(model, "sub_module.0")
    assert parent == model.sub_module
    assert name == "0"
    assert isinstance(getattr(parent,name), nn.Linear)

# Tests for FouRAInjectedLinear
@pytest.mark.parametrize("N_in, N_out, rank, transform",[
    (32, 64, 8, "dct"), (16,16,4,"fft"), (8,12,2,"none")
])
def test_foura_injected_linear_init_and_forward(N_in, N_out, rank, transform):
    base_linear = nn.Linear(N_in, N_out)
    cfg = FouRAConfig(rank=rank, transform_type=transform)
    
    # Test requires_grad is False for base_linear parameters
    with torch.no_grad(): # Ensure base_linear params are not modified during init
        base_linear_original_weight = base_linear.weight.clone()
        base_linear_original_bias = base_linear.bias.clone() if base_linear.bias is not None else None

    injected_linear = FouRAInjectedLinear(base_linear, cfg)

    # Check base layer parameters are frozen
    for param in injected_linear.base.parameters():
        assert not param.requires_grad
    
    # Check FouRA module parameters are trainable by default (within FouRA module itself)
    # This test assumes FouRA parameters are requires_grad=True by default
    for param in injected_linear.foura.parameters():
        assert param.requires_grad

    assert injected_linear.in_features == N_in
    assert injected_linear.out_features == N_out
    assert isinstance(injected_linear.foura, FouRA)
    assert injected_linear.foura.down.in_features == (N_in // 2 + 1 if transform == "fft" else N_in)

    x = torch.randn(2, 10, N_in) # Batch=2, SeqLen=10
    output = injected_linear(x)
    assert output.shape == (2, 10, N_out)

    # Check that base_linear weights weren't changed by injection
    assert torch.equal(injected_linear.base.weight, base_linear_original_weight)
    if base_linear_original_bias is not None:
        assert torch.equal(injected_linear.base.bias, base_linear_original_bias)
    else:
        assert injected_linear.base.bias is None

def test_foura_injected_linear_type_error():
    with pytest.raises(TypeError):
        FouRAInjectedLinear(nn.ReLU(), FouRAConfig()) # type: ignore

# Tests for get_foura_model
def test_get_foura_model_injection(dummy_linear_model):
    model = dummy_linear_model
    # Use target_modules=[""] to match all Linear layers since "" is in every layer name
    # and get_foura_model also checks isinstance(module, nn.Linear).
    cfg = FouRAConfig(rank=8, target_modules=[""])

    num_linear_before = sum(isinstance(m, nn.Linear) for m in model.modules())
    assert num_linear_before == 3 # 0, 2, and sub_module.0

    modified_model = get_foura_model(model, cfg)
    
    # Check parameters' requires_grad status
    has_classifier = hasattr(modified_model, "classifier") and isinstance(modified_model.classifier, nn.Module)
    # Properly get classifier parameters if the classifier exists and is a Module
    classifier_params = []
    if has_classifier:
        # Ensure classifier itself is a module before trying to get its parameters
        if isinstance(modified_model.classifier, nn.Module):
            classifier_params = list(modified_model.classifier.parameters())
        else:
            # Handle cases where model.classifier might not be an nn.Module (e.g. a function)
            # For this test, dummy_linear_model doesn't have a classifier, so this path isn't critical
            pass 

    for name, param in modified_model.named_parameters():
        is_base_param = ".base." in name
        is_foura_param = ".foura." in name
        is_adapter_proj_param = ".adapter_projection." in name
        is_classifier_param = any(param is p for p in classifier_params) # Check identity

        if is_base_param:
            assert not param.requires_grad, f"Base parameter {name} should be frozen."
        elif is_foura_param or is_adapter_proj_param:
            assert param.requires_grad, f"FouRA/Adapter parameter {name} should be trainable."
        elif is_classifier_param:
            assert param.requires_grad, f"Classifier parameter {name} should be trainable."
        else:
            # Parameters of non-wrapped layers (e.g., nn.ReLU has no params)
            # or other original model parts should be frozen by model.requires_grad_(False)
            assert not param.requires_grad, f"Other parameter {name} ({param.shape}) should be frozen."

    # Check if injection happened
    assert isinstance(modified_model[0], FouRAInjectedLinear)
    assert isinstance(modified_model[2], FouRAInjectedLinear)
    assert isinstance(modified_model.sub_module[0], FouRAInjectedLinear)
    
    # Check original non-linear layers are untouched
    assert isinstance(modified_model[1], nn.ReLU)

    # Check config is attached
    assert hasattr(modified_model, "foura_config")
    assert modified_model.foura_config["rank"] == 8

def test_get_foura_model_no_match_value_error(dummy_linear_model):
    model = dummy_linear_model
    cfg = FouRAConfig(target_modules=["NonExistentModule"])
    with pytest.raises(ValueError, match="found no matching Linear layers"):
        get_foura_model(model, cfg)

def test_get_foura_model_target_specific_layer(simple_model_with_target_name):
    model = simple_model_with_target_name
    cfg = FouRAConfig(rank=4, target_modules=["vit_block.attention.0"])
    
    modified_model = get_foura_model(model, cfg)

    assert isinstance(modified_model.vit_block.attention[0], FouRAInjectedLinear)
    assert isinstance(modified_model[0], nn.Linear) # First linear should not be wrapped
    assert isinstance(modified_model.vit_block[0], nn.Linear) # vit_block.0 should not be wrapped 