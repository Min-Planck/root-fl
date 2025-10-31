from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
import flwr as fl
import torch 
import numpy as np 
import yaml 
from easydict import EasyDict

from src.utils import get_train_data, set_seed, get_configs, get_model, get_algorithm, get_parameters
from src.client import BaseClient, SimpleClientManager

# ----------- CHANGE HERE -------------
ALGO = 'fedntd'
MODEL = 'resnet18'
DATASET = 'cifar10'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}
# -------------------------------------

with open("./config/general_conf.yaml") as f: 
    cfg = EasyDict(yaml.safe_load(f))

general_cfg = cfg[ALGO]

with open("./config/model_conf.yaml") as f:
    m_cfg = EasyDict(yaml.safe_load(f))

m_cfg = m_cfg[MODEL]
set_seed(general_cfg.seed_value)
ids, dist, trainloaders, testloader, client_dataset_ratio = get_train_data(
    dataset_name=DATASET,
    num_clients=general_cfg.num_clients,
    batch_size=general_cfg.batch_size, 
    fractions=general_cfg.partition_fraction,
    alphas=general_cfg.partition_alpha
)

algorithm_config = get_configs(
    cfg=general_cfg,
    dist=dist,
    algo_name=ALGO
    )
    
def client_fn(context: Context) -> BaseClient:
    cid = int(context.node_config["partition-id"])
    is_moon_type = True if ALGO == 'moon' else False
    net = get_model(
        model_name=MODEL,
        num_channels=m_cfg.channels,
        img_size=m_cfg.img_size,
        num_classes=m_cfg.num_classes,
        hidden_size=m_cfg.hidden_size if 'hidden_size' in m_cfg.keys() else None,
        moon_type=is_moon_type
    )
    net.to(DEVICE)
    trainloader = trainloaders[int(cid)]  
    num_classes = sum(v is not None and v > 0 for v in dist[cid].values())
    return BaseClient(cid, ALGO, net, trainloader, num_classes, general_cfg.local_training_epochs, DEVICE).to_client()

is_moon_type = True if ALGO == 'moon' else False 
net_ = get_model(
        model_name=MODEL,
        num_channels=m_cfg.channels,
        img_size=m_cfg.img_size,
        num_classes=m_cfg.num_classes,
        hidden_size=m_cfg.hidden_size if 'hidden_size' in m_cfg.keys() else None,
        moon_type=is_moon_type
    )

current_parameters = ndarrays_to_parameters(get_parameters(net_)) 

algo = get_algorithm(ALGO) 
strategy = algo(
    exp_name=f"{ALGO}_{DATASET}_{MODEL}",
    net=net_,
    num_rounds=general_cfg.num_rounds,
    num_clients=general_cfg.num_clients,
    testloader=testloader,
    algorithm_config=algorithm_config,
    learning_rate=general_cfg.learning_rate,
    current_parameters=current_parameters
)

print(f"Starting Federated Learning with {ALGO} on {DATASET} dataset")

fl.simulation.start_simulation(
            client_fn           = client_fn,
            num_clients         = general_cfg.num_clients,
            config              = fl.server.ServerConfig(num_rounds=general_cfg.num_rounds),
            strategy            = strategy,
            client_manager      = SimpleClientManager(),
            client_resources    = client_resources
        )