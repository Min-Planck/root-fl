import flwr as fl 
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.utils import set_parameters, get_parameters 
from .utils import fit_handler

class BaseClient(fl.client.NumPyClient):
    def __init__(self,
                 cid,                  
                 algo_name, 
                 net, 
                 trainloader, 
                 num_classes,
                 local_train_epochs, 
                 device):
         
        self.cid = cid
        self.local_train_epochs = local_train_epochs
        self.algo_name = algo_name
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.client_control = None
        self.num_classes = num_classes

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        config = {**config, **{"epochs": self.local_train_epochs,"num_classes": self.num_classes, "device": self.device}}
        set_parameters(self.net, parameters)
        metrics = fit_handler(algo_name=self.algo_name, cid=self.cid, net=self.net, trainloader=self.trainloader, config=config, client_control=self.client_control, parameters=parameters)
        
        if self.algo_name == "fedcls":
            metrics["num_classes"] = self.num_classes
        if self.algo_name == "scaffold":
            self.client_control = metrics["client_control"]
            params_obj = metrics['params']
            client_params_news = parameters_to_ndarrays(params_obj)
            _, _ = metrics.pop("params", None), metrics.pop("client_control", None)
        else: 
            client_params_news = get_parameters(self.net)

        metrics = {k: v for k, v in metrics.items() if v is not None}
        return client_params_news, len(self.trainloader.sampler), metrics    
    
    def evaluate(self, parameters, config):
        return None