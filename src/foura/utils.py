import torch

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0:
            continue
        all_param += num_params
        if param.requires_grad:
            print(f"► Trainable: {name} - Shape: {param.shape}")
            trainable_params += num_params
        else:
            print(f"Frozen: {name} - Shape: {param.shape}")
    print(
        f"\nTrainable params: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}%)"
        f"\nAll params: {all_param:,d}"
    ) 