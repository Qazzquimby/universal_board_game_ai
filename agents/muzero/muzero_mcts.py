import random
from typing import List, Union

import torch
from jaxtyping import Float

from agents.muzero.muzero_net import MuZeroNet
from algorithms.mcts import Edge, MCTSNodeWithState, MCTSNode
from environments.base import StateWithKey, ActionType
from algorithms.sampling import ProgWidener


# Currently assuming alternating between 2 players
# In future, better to predict the current player at a state from the model
# Or at least handle more than 2 players


def get_next_player(current_player_index: int, num_players: int = 2) -> int:
    return (current_player_index + 1) % num_players


class MuZeroObservedRootNode(MCTSNodeWithState):
    """Root node from the player's observation"""

    def __init__(
        self,
        state_with_key: StateWithKey,
        actions: List[ActionType],
        current_player_index: int,
        network: "MuZeroNet",
    ):
        super().__init__(state_with_key=state_with_key)
        self.network = network
        self.action_tokens = network.tokenize_actions(actions)
        self.player_idx = current_player_index  # uses property current_player_index
        self.revelations = []
        (
            mu,
            log_var,
        ) = network.get_root_state_observation_to_revealed_latent_sampler_params(
            state=state_with_key.state
        )
        self.widener = ProgWidener(mu=mu, log_var=log_var)

    def get_revelation(self):
        if self.revelations:
            existing_children = torch.stack([rev.latent for rev in self.revelations])
            new_revelation = self.widener.widen_if_needed(existing_children)
        else:
            new_revelation = self.widener.widen_if_needed()

        if new_revelation is not None:
            new_revelation_node = MuZeroRevealedRootNode(
                network=self.network,
                observed_root=self,
                latent=new_revelation,
                action_tokens=self.action_tokens,
            )
            self.revelations.append(new_revelation_node)
            return new_revelation_node
        else:
            return random.choice(self.revelations)


class MuZeroRevealedRootNode(MCTSNode):
    """Possible revelation of the root node given hidden info"""

    edges: List["MuZeroEdge"]  # for type hint

    def __init__(
        self,
        observed_root: MuZeroObservedRootNode,
        latent: Float[torch.Tensor, "1 emb_dim"],
        action_tokens: Float[torch.Tensor, "1 num_actions emb_dim"],
        network: MuZeroNet,
    ):
        super().__init__()
        self.observed_root = observed_root
        self.latent = latent
        self.action_tokens = action_tokens
        prior = network.root_state_observation_to_policy(
            state_latent=self.latent, action_token=action_tokens
        )
        (
            successor_sampler_mu,
            successor_sampler_log_var,
        ) = network.state_latent_and_action_to_successor_latent_sampler(
            state_latent=self.latent, action_token=action_tokens
        )
        for i in range(action_tokens.shape[1]):
            edge = MuZeroEdge(
                network=network,
                prior=prior[0][i].item(),
                successor_sampler_mu=successor_sampler_mu[0][i],
                successor_sampler_log_var=successor_sampler_log_var[0][i],
                next_player_index=get_next_player(self.current_player_index),
            )
            self.edges[i] = edge

    @property
    def current_player_index(self):
        return self.observed_root.current_player_index


class MuZeroInnerNode:
    def __init__(
        self, latent: torch.Tensor, current_player_index: int, network: "MuZeroNet"
    ):
        self.latent = latent
        self.current_player_index = current_player_index
        self.edges: List[MuZeroEdge] = self._init_actions(network=network)

    def _init_actions(self, network):
        action_tokens, priors = network.get_available_actions_and_priors(
            state_latent=self.latent
        )
        edges = []
        for action_token, prior in zip(action_tokens.tolist(), priors.tolist()):
            (
                successor_sampler_mu,
                successor_sampler_log_var,
            ) = network.get_state_latent_to_successor_latent_sampler_params(
                state_latent=self.latent, action_token=action_token
            )
            edge = MuZeroEdge(
                network=network,
                prior=prior,
                successor_sampler_mu=successor_sampler_mu,
                successor_sampler_log_var=successor_sampler_log_var,
                next_player_index=get_next_player(self.current_player_index),
            )
            edges.append(edge)
        return edges


MuZeroNode = Union[MuZeroObservedRootNode, MuZeroInnerNode]


class MuZeroEdge(Edge):
    def __init__(
        self,
        prior: float,
        network: "MuZeroNet",
        successor_sampler_mu: torch.Tensor,
        successor_sampler_log_var: torch.Tensor,
        next_player_index: int,
    ):
        super().__init__(prior=prior)
        self.network = network
        self.child_nodes: List["MuZeroInnerNode"] = []
        self.next_player_index = next_player_index

        self.widener = ProgWidener(
            mu=successor_sampler_mu, log_var=successor_sampler_log_var
        )

    def get_child_node(self):
        new_successor_latent = self.widener.widen_if_needed(
            existing_children=torch.stack([node.latent for node in self.child_nodes])
        )
        if new_successor_latent is not None:
            new_child_node = MuZeroInnerNode(
                latent=new_successor_latent,
                current_player_index=self.next_player_index,
                network=self.network,
            )
            self.child_nodes.append(new_child_node)
            return new_child_node
        else:
            return random.choice(self.child_nodes)
