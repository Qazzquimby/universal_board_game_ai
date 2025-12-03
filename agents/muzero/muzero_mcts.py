import random
from typing import List, Union

import torch

from agents.muzero.muzero_net import MuZeroNet
from algorithms.mcts import Edge, MCTSNodeWithState, MCTSNode
from environments.base import StateWithKey, ActionType
from algorithms.sampling import ProgWidener


class MuZeroObservedRootNode(MCTSNodeWithState):
    """Root node from the player's observation"""

    def __init__(
        self,
        state_with_key: StateWithKey,
        actions: List[ActionType],
        player_idx: int,
        network: "MuZeroNet",
    ):
        super().__init__(state_with_key=state_with_key)
        self.network = network
        self.action_tokens = network.tokenize_actions(actions)
        self.player_idx = player_idx
        self.revelations = []
        (
            mu,
            log_var,
        ) = network.get_root_state_observation_to_revealed_latent_sampler_params(
            state=state_with_key.state
        )
        self.widener = ProgWidener(mu=mu, log_var=log_var)

    def get_revelation(self):
        new_revelation = self.widener.widen_if_needed(
            existing_children=torch.stack([rev.latent for rev in self.revelations])
        )
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
        latent: torch.Tensor,
        action_tokens: torch.Tensor,
        network: "MuZeroNet",
    ):
        super().__init__()
        self.observed_root = observed_root
        self.latent = latent
        self.action_tokens = action_tokens

        for action_token in action_tokens.tolist():
            prior = network.get_action_prior_from_state_latent(
                state_latent=self.latent, action_token=action_token
            )
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
            )
            self.edges.append(edge)

    @property
    def player_idx(self):
        return self.observed_root.player_idx


class MuZeroInnerNode:
    def __init__(self, latent: torch.Tensor, player_idx: int, network: "MuZeroNet"):
        self.latent = latent
        self.player_idx = player_idx
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
    ):
        super().__init__(prior=prior)
        self.network = network
        self.child_nodes: List["MuZeroInnerNode"] = []

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
                network=self.network,
            )
            self.child_nodes.append(new_child_node)
            return new_child_node
        else:
            return random.choice(self.child_nodes)
