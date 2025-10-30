from ..fl_common_import import *
from ..fedavg.fedavg import FedAvg

class FedNTD(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = self.algorithm_config['tau']
        self.beta = self.algorithm_config['beta']

    def __repr__(self) -> str:
        return "FedNTD"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config = {"tau": self.tau, "beta": self.beta, "learning_rate": self.learning_rate}

        return [(client, FitIns(parameters, config)) for client in clients]  