from ..fl_common_import import *
from ..fedavg.fedavg import FedAvg

class FedDisco(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.algorithm_config['alpha'] 
        self.beta = self.algorithm_config['beta']
        self.dk = self.algorithm_config['dk'] 

    def __repr__(self) -> str:
        return "FedDisco"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        pk, cummulate = {}, 0 
        for _, fit_res in results: 
            cid = fit_res.metrics['id']
            value = F.relu(torch.tensor(fit_res.num_examples - self.alpha * self.dk[cid] + self.beta))
            pk[cid] = np.array(value)
            cummulate += pk[cid] 
        
        pk = {cid: pk[cid] / cummulate for cid in pk.keys()}

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), pk[fit_res.metrics['id']])for _, fit_res in results]
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated