from ..fl_common_import import *
from ..fedavg.fedavg import FedAvg

class MOON(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = self.algorithm_config['temperature']

    def __repr__(self):
        return "MOON"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate, "temperature": self.temperature}

        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs