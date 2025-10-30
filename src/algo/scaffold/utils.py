import numpy as np  
import torch
from torch import nn
from typing import List
from collections import OrderedDict

def initialize_control_from_weights(weights: List[np.ndarray]) -> List[np.ndarray]:
    """Initialize control variates as zeros with same shape as model weights"""
    return [np.zeros_like(w) for w in weights]

def parameters_to_torch_dict(parameters: List[np.ndarray], reference_model) -> OrderedDict:
    """Convert numpy parameters to PyTorch state dict format"""
    state_dict_keys = list(reference_model.state_dict().keys())
    if len(parameters) != len(state_dict_keys):
        raise ValueError(f"Parameters length {len(parameters)} doesn't match model state dict length {len(state_dict_keys)}")
    
    torch_dict = OrderedDict()
    for key, param in zip(state_dict_keys, parameters):
        torch_dict[key] = torch.from_numpy(param)
    return torch_dict

def torch_dict_to_parameters(torch_dict: OrderedDict) -> List[np.ndarray]:
    """Convert PyTorch state dict to numpy parameters"""
    return [tensor.cpu().numpy() for tensor in torch_dict.values()]

def initialize_control_from_model(reference_model) -> List[np.ndarray]:
    """Initialize control variates as zeros with same structure as model parameters"""
    control_dict = OrderedDict()
    for name, param in reference_model.state_dict().items():
        control_dict[name] = torch.zeros_like(param)
    return torch_dict_to_parameters(control_dict)

def train_scaffold(
    net,
    trainloader,
    learning_rate,
    epochs,
    device, 
    client_control_old,
    server_control,
    client_control
):
    from src.utils import get_parameters
    from flwr.common import ndarrays_to_parameters
    
    model_state_dict = net.state_dict()
    model_keys = list(model_state_dict.keys())
    model_params = get_parameters(net)
    
    if len(client_control_old) != len(model_params) or len(server_control) != len(model_params):
        raise ValueError(f"Control arrays length mismatch: model={len(model_params)}, "
                        f"client_control={len(client_control_old)}, server_control={len(server_control)}")
    
    correction_dict = {}
    named_params = dict(net.named_parameters())
    
    for i, (key, c_i, c_s, model_param) in enumerate(zip(model_keys, client_control_old, server_control, model_params)):
        if c_i.shape != model_param.shape or c_s.shape != model_param.shape:
            raise ValueError(f"Shape mismatch at parameter {i} ({key})")
        
        if key in named_params and named_params[key].requires_grad:
            correction = torch.tensor(c_i - c_s, dtype=torch.float32, device=device)
            correction_dict[key] = correction
    
    initial_weights = get_parameters(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    net.train() 
    total_loss, correct, total = 0.0, 0, 0
    num_batches = 0
    
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            
            with torch.no_grad():
                for param_name, param in net.named_parameters():
                    if param.grad is not None and param_name in correction_dict:
                        corr = correction_dict[param_name]
                        if param.grad.shape == corr.shape:
                            param.grad -= corr
                        else:
                            print(f"Shape mismatch at param {param_name}: grad={param.grad.shape}, corr={corr.shape}")
                    elif param.grad is not None:
                        print(f"Skipping correction for {param_name} (not in correction_dict or not trainable)")
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            num_batches += 1

        updated_weights = get_parameters(net)

        
        # Calculate control update: Δc_i = -c_s + (w0 - wT)/(Kη)
        K = num_batches  # Number of local steps
        eta = learning_rate  # Learning rate
        control_update = [
            (w0 - wT) / (K * eta) - c_s
            for w0, wT, c_s in zip(initial_weights, updated_weights, server_control)
        ]
        
        # Update client control: c_i^{new} = c_i^{old} + Δc_i
        client_control = [
            c_old + delta 
            for c_old, delta in zip(client_control_old, control_update)
        ]
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total,
        'params': ndarrays_to_parameters(
                updated_weights + control_update
            ),
        'client_control': client_control
        }
