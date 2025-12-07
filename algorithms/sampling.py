import math

import einops
import torch
from typing import Optional

from jaxtyping import Float


class ProgWidener:
    def __init__(
        self,
        mu: Float[torch.Tensor, "1 emb"],
        log_var: Float[torch.Tensor, "1 emb"],
        min_distance_for_child: float = 2.0,
        base_accesses_per_widen: int = 10,
    ):
        self.mu = mu
        self.log_var = log_var
        self.min_distance_for_child = min_distance_for_child
        self.base_accesses_per_widen = base_accesses_per_widen

        self.num_widens = 0
        self.num_accesses = 0

    def widen_if_needed(
        self, existing_children: Optional[Float[torch.Tensor, "1 action emb"]] = None
    ) -> Optional[Float[torch.Tensor, "1 emb"]]:
        self.num_accesses += 1
        needed_widens = self._get_needed_widens()
        if needed_widens > self.num_widens or existing_children is None:
            return self._widen(
                existing_children=existing_children,
            )
        return None

    def _get_needed_widens(self):
        if self.num_accesses <= self.base_accesses_per_widen:
            return 1

        return (
            math.floor(math.log2(self.num_accesses / self.base_accesses_per_widen)) + 2
        )

    def _widen(
        self, existing_children: Optional[Float[torch.Tensor, "1 action emb"]] = None
    ) -> Optional[Float[torch.Tensor, "1 emb"]]:
        self.num_widens += 1

        sample = take_sample(mu=self.mu, log_var=self.log_var)
        _batch, _emb = sample.shape

        if existing_children is None:
            return sample
        _1, _action, _emb = existing_children.shape
        assert _1 == 1

        distances = self._get_sample_distances(
            existing_children=existing_children, sample=sample
        )
        min_distance = torch.min(distances)
        if min_distance >= self.min_distance_for_child:
            return sample
        else:
            return None

    def _get_sample_distances(
        self,
        existing_children: Float[torch.Tensor, "1 action emb"],
        sample: Float[torch.Tensor, "1 emb"],
    ):
        _1, _action, _emb = existing_children.shape
        assert _1 == 1
        _1, _emb = sample.shape

        stacked_sample = einops.repeat(sample, "1 emb -> 1 action emb", action=_action)
        var = torch.exp(self.log_var)
        distances = torch.sqrt(
            torch.sum(((existing_children - stacked_sample) ** 2) / var)
        )
        return distances


def take_sample(
    mu: Float[torch.Tensor, "batch emb"], log_var: Float[torch.Tensor, "batch emb"]
) -> Float[torch.Tensor, "batch emb"]:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
