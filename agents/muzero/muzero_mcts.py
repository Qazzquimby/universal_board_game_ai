import random
from typing import List, Dict

import torch

from agents.muzero.muzero_agent import ProgWidener
from agents.muzero.muzero_net import MuZeroNet
from algorithms.mcts import Edge, MCTSNodeWithState
from environments.base import StateWithKey


class MuZeroEdge(Edge):  # ABC
    def __init__(self, prior: float):
        super().__init__(prior)
        # A single edge can lead to multiple outcomes (child nodes) due to stochastic dynamics.
        self.child_nodes: List["MuZeroNode"] = []


class MuZeroObservedRootNode(MCTSNodeWithState):
    """Root node from the player's observation"""

    def __init__(
        self,
        state_with_key: StateWithKey,
        player_idx: int,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ):
        super().__init__(state_with_key=state_with_key)
        self.player_idx = player_idx
        self.revelations = []
        self.widener = ProgWidener(mu=mu, log_var=log_var)

    def get_revelation(self):
        new_revelation = self.widener.widen_if_needed(
            existing_children=torch.stack([rev.latent for rev in self.revelations])
        )
        if new_revelation is not None:
            new_revelation_node = MuZeroRevealedRootNode(
                player_idx=self.player_idx,
                latent=new_revelation,
            )
            self.revelations.append(new_revelation_node)
            return new_revelation_node
        else:
            return random.choice(self.revelations)

    @classmethod
    def get_sampler_params(cls):
        pass  # todo? Better way of handling this?


class MuZeroRevealedRootNode:
    """Possible revelation of the root node given hidden info"""

    def __init__(self, player_idx: int, latent: torch.Tensor):
        self.player_idx = player_idx
        self.latent = latent
        self.edges: Dict[int, MuZeroRootEdge] = {}


class MuZeroRootEdge:
    class MuZeroInnerEdge:
        pass
