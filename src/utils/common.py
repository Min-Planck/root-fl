import numpy as np
from collections import OrderedDict
from typing import List, Dict
import random
import math
import torch
import torch.nn as nn
from torch.optim import SGD

from src.algo import get_moon_model, MoonTypeModel, FedAdp, FedAvg, FedCLS, FedDisco, FedImp, FedNTD, MOON, Scaffold
from src.models import ResNet18, ResNet34, ResNet50, ResNet101, CNN_Text, MLP

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds set to {seed_value}")

def normalize_distribution(distribution):
    total = sum(distribution.values())
    return [count / total for count in distribution.values()]

def compute_uniform_distribution(num_classes):
    return [1 / num_classes] * num_classes

def get_distribution_diff_from_uniform(distribution, num_clients): 
    from src.utils.distance import kl_divergence
    dk = {} 
    num_classes = len(distribution[0]) 

    uniform_dist = compute_uniform_distribution(num_classes=num_classes)

    for client_id in range(num_clients): 
        client_dist = distribution[client_id]
        normalized_client_dist = normalize_distribution(client_dist)
        diff = kl_divergence(p=normalized_client_dist, q=uniform_dist)
        dk[client_id] = diff
    
    return dk

def get_model(model_name, 
              num_channels, 
              img_size, 
              num_classes, 
              hidden_size=None, 
              moon_type=False):

    if moon_type: 
        return get_moon_model(
            model_name,
            num_channels=num_channels,
            im_size=img_size,
            num_classes=num_classes,
            hidden_size=hidden_size
        )
    
    if model_name == 'resnet50': 
        model = ResNet50(num_channel=num_channels, 
                         num_classes=num_classes)
    elif model_name == 'resnet18': 
        model = ResNet18(num_channel=num_channels, 
                         num_classes=num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_channel=num_channels, 
                         num_classes=num_classes)
    elif model_name == 'resnet101':
        model = ResNet101(num_channel=num_channels, 
                         num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_channel=num_channels, 
                         num_classes=num_classes)
    elif model_name == 'cnn_text':
        model = CNN_Text()
    elif model_name == 'mlp': 
        model = MLP(
            num_channels=num_channels,
            im_size=img_size,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

    return model

def get_configs(cfg, dist, algo_name): 
    
    algorithm_config = {}
    client_config = {}

    if algo_name == 'fedcls': 
        all_classes = len(dist[0])
        
        algorithm_config['alpha'] = cfg.alpha
        algorithm_config['all_classes'] = all_classes

    elif algo_name == 'fedadp':
        algorithm_config['alpha'] = cfg.alpha

    elif algo_name == 'feddisco': 
        algorithm_config['alpha'] = cfg.alpha
        algorithm_config['beta'] = cfg.beta

        dk = get_distribution_diff_from_uniform(dist, cfg.num_clients)
        algorithm_config = {**algorithm_config, **{'dk': dk}}

    elif algo_name == 'fedimp':
        entropies = [compute_entropy(dist[i]) for i in range(cfg.num_clients)]
        algorithm_config = {**algorithm_config, **{'entropies': entropies}}

    elif algo_name == 'fedntd': 
        algorithm_config['tau'] = cfg.tau
        algorithm_config['beta'] = cfg.beta
    elif algo_name == 'moon':
        algorithm_config['temperature'] = cfg.temperature

    return algorithm_config

def get_algorithm(algo_name): 
    if algo_name == 'fedcls':
        return FedCLS
    elif algo_name == 'fedadp':
        return FedAdp
    elif algo_name == 'feddisco':
        return FedDisco
    elif algo_name == 'fedimp':
        return FedImp
    elif algo_name == 'fedntd':
        return FedNTD
    elif algo_name == 'moon':
        return MOON
    elif algo_name == 'fedavg':
        return FedAvg
    elif algo_name == 'scaffold':
        return Scaffold
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")