# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        #straggler_schedule: np.ndarray,
     partition_id

    ):  # pylint: disable=too-many-arguments
        #self.net = net
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        #self.straggler_schedule = straggler_schedule,
        self.client_id=partition_id


    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net.
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        """
        """Return the parameters of the encoder and classifier."""
        return [
            val.cpu().numpy() for _, val in self.encoder.state_dict().items()
        ] + [
            val.cpu().numpy() for _, val in self.classifier.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones.
        #Combines the keys of the state dictionary with the new parameter values into a list of key-value pairs.
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
"""
        """Set the parameters of the encoder and classifier."""
        print("Parameters structure:")
        for i, param in enumerate(parameters):
          print(f"Parameter {i}: Shape = {param.shape}")

        encoder_params = zip(self.encoder.state_dict().keys(),parameters[:len(self.encoder.state_dict())])
        classifier_params = zip(self.encoder.state_dict().keys(),parameters[len(self.encoder.state_dict()):])

        encoder_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in  encoder_params})
        classifier_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in classifier_params })
      

        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.classifier.load_state_dict(classifier_state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)

        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        '''
        if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
            
            if config["drop_client"]:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.trainloader),
                    {"is_straggler": True},
                )
        '''
        #else:
        num_epochs = self.num_epochs

        train_gpaf(
            self.net,
            self.trainloader,
            self.device,
            self.client_id,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
           
        )

        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test_gpaf(self.net, self.valloader, self.device)
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

