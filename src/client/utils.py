import os
import copy
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from src.utils import train, get_parameters
from src.algo import train_moon, train_scaffold

MOON_SAVE_DIR = '/moon_save_point/'
DEFAULT_METRICS = {
    "loss": None,
    "accuracy": None,
    "f1": None,
    "recall": None,
    "precision": None,
    "id": None, 
    "num_classes": None, 
    "client_control": None
}

def fit_handler(algo_name, cid, config, net, trainloader, client_control=None, parameters=None):
    """
    Handler function to return the metrics based on the algorithm name.
    
    Args:
        algo_name (str): Name of the algorithm.
        cid (str): Client ID.
        net (torch.nn.Module): The neural network model to be trained.
        trainloader (DataLoader): DataLoader for the training dataset.
        config (dict): Configuration parameters for the algorithm.
        client_control (list, optional): Client control variates for Scaffold algorithm. Defaults to None.
        parameters (Parameters, optional): Model parameters. Defaults to None.
    Returns:
        dict: A dictionary containing the metrics such as loss, accuracy, and necessary metrics for some algorithm.
    """
    if algo_name in ["fedavg", "feddisco", "fedcls", "fedadp", "fedimp"]:
        res_metrics = train(net, trainloader, DEVICE=config['device'], learning_rate=config["learning_rate"], epochs=config["epochs"])
    elif algo_name == "fedntd":
        res_metrics = train(net, trainloader, DEVICE=config['device'], learning_rate=config["learning_rate"], epochs=config["epochs"], use_ntd_loss=True, tau=config["tau"], beta=config["beta"])
    elif algo_name == "moon": 
        save_dir = os.path.join(MOON_SAVE_DIR, f"client_{cid}")
        pre_round_net = copy.deepcopy(net)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else: 
            prev_net_path = os.path.join(save_dir, "prev_net.pt")
            if os.path.exists(prev_net_path):
                pre_round_net.load_state_dict(torch.load(prev_net_path))
        
        global_net = copy.deepcopy(net)
        
        _, loss, acc = train_moon(
            net,
            global_net,
            pre_round_net,
            trainloader,
            lr=config["learning_rate"],
            temperature=config["temperature"],
            device=config["device"],
            epochs=config["epochs"],
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            torch.save(net.state_dict(), os.path.join(save_dir, "prev_net.pt"))
            
        res_metrics = {
            "loss": loss,
            "accuracy": acc
        }
    elif algo_name == "scaffold":
        if isinstance(parameters, list):
            parameters = ndarrays_to_parameters(parameters)
        full_params = parameters_to_ndarrays(parameters)
        

        model_params = get_parameters(net)
        num_model_params = len(model_params)
        
        expected_length = 3 * num_model_params
        if len(full_params) != expected_length:
            num_model_params = len(full_params) // 3
        
        model_weights = full_params[:num_model_params]
        server_control = full_params[num_model_params:2*num_model_params]
        client_control_old = full_params[2*num_model_params:3*num_model_params]
        
        if client_control is None:
            client_control = [np.zeros_like(w) for w in model_weights]

        res_metrics = train_scaffold(
            net,
            trainloader,
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            device=config["device"],
            client_control_old=client_control_old,
            server_control=server_control,
            client_control=client_control
        )
    elif algo_name == 'fedaaw': 
        res_metrics = train(net, trainloader, DEVICE=config['device'], learning_rate=config["learning_rate"], epochs=config["epochs"], get_grad_norm=True)
    
    return {**DEFAULT_METRICS, **res_metrics, **{"id": int(cid)}}