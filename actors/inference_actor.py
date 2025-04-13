from typing import Tuple, List, Dict
import numpy as np
import ray
import torch

from environments.base import StateType, BaseEnvironment # For type hints and network init
from models.networks import AlphaZeroNet # Import the network class
from core.config import AlphaZeroConfig # For network init args

# Helper to get a dummy env for network init if needed
from factories import get_environment
from core.config import EnvConfig

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class InferenceActor:
    """
    A Ray actor dedicated to running batch inference on a neural network (ideally on a GPU).
    """
    def __init__(
        self,
        initial_network_state: Dict,
        env_config: EnvConfig, # Need env config to init network
        agent_config: AlphaZeroConfig # Need agent config to init network
        ):
        # Determine device based on Ray resource allocation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"InferenceActor initializing on device: {self.device}")

        # Need a dummy env instance to determine network dimensions
        dummy_env = get_environment(env_config)

        self.network = AlphaZeroNet(
            dummy_env,
            hidden_layer_size=agent_config.hidden_layer_size,
            num_hidden_layers=agent_config.num_hidden_layers
        )
        if initial_network_state:
            # Ensure state dict tensors are loaded onto the correct device
            loaded_state_dict = {k: v.to(self.device) for k, v in initial_network_state.items()}
            self.network.load_state_dict(loaded_state_dict)

        self.network.to(self.device)
        self.network.eval()
        print(f"InferenceActor network loaded on device: {self.device}")


    def predict_batch(
        self, state_dicts: List[StateType]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs batched inference on a list of state dictionaries.
        Input states are expected to be raw dictionaries.
        Returns results as CPU numpy arrays/floats.
        """
        if not state_dicts:
            return [], []

        # Network's internal predict_batch handles flattening, device moving, and conversion back to CPU numpy
        # Ensure the network's predict_batch method correctly uses self.device
        policy_list, value_list = self.network.predict_batch(state_dicts)
        return policy_list, value_list

    def update_weights(self, network_state: Dict):
        """Updates the network weights."""
        try:
            # Ensure state dict tensors are loaded onto the correct device
            loaded_state_dict = {k: v.to(self.device) for k, v in network_state.items()}
            self.network.load_state_dict(loaded_state_dict)
            self.network.eval()
            print(f"InferenceActor weights updated on device: {self.device}")
        except Exception as e:
            print(f"InferenceActor Error loading network state: {e}")

    def get_device(self) -> str:
        """Returns the device the actor is running on."""
        return str(self.device)
