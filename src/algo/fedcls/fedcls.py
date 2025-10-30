from ..fl_common_import import *
from ..fedavg.fedavg import FedAvg

class FedCLS(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.algorithm_config['alpha'] 
        self.all_classes = self.algorithm_config['all_classes']

    def __repr__(self) -> str:
        return "FedCLS"


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        examples = [fit_res.num_examples for _, fit_res in results]
        weights_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * (1 - self.alpha) + (self.alpha * fit_res.metrics["num_classes"] / self.all_classes))
                            for _, fit_res in results]

        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated