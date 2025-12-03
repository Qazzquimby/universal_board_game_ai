import math

import einops
import torch
from typing import Optional


class ProgWidener:
    def __init__(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
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
        self, existing_children: torch.Tensor
    ) -> Optional[torch.Tensor]:
        self.num_accesses += 1
        needed_widens = self._get_needed_widens()
        if needed_widens > self.num_widens:
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

    def _widen(self, existing_children: torch.Tensor) -> Optional[torch.Tensor]:
        self.num_widens += 1

        sample = take_sample(mu=self.mu, log_var=self.log_var)
        distances = self._get_sample_distances(
            existing_children=existing_children, sample=sample
        )
        min_distance = torch.min(distances)
        if min_distance >= self.min_distance_for_child:
            return sample
        else:
            return None

    def _get_sample_distances(
        self, existing_children: torch.Tensor, sample: torch.Tensor
    ):
        stacked_sample = einops.repeat(
            sample, "emb -> h emb", h=existing_children.shape[0]
        )
        var = torch.exp(self.log_var)
        distances = torch.sqrt(
            torch.sum(((existing_children - stacked_sample) ** 2) / var)
        )
        return distances


def take_sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
