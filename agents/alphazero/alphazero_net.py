from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.base import BaseEnvironment, StateType, StateWithKey, ActionType
from models.networks import BaseTokenizingNet


class AlphaZeroNet(BaseTokenizingNet):
    """
    A generic AlphaZero-style network that uses a transformer to process
    the game state represented as a dictionary of DataFrames.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(env=env, embedding_dim=embedding_dim)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.game_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # --- Heads ---
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())
        policy_input_dim = embedding_dim + embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.cache = {}

    def _get_state_embedding_and_value(
        self, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a single state to get its embedding and value."""
        tokens = self._state_to_tokens(state)  # (1, num_tokens, dim)

        # Prepend game token
        game_token = self.game_token.expand(1, -1, -1)
        sequence = torch.cat([game_token, tokens], dim=1)

        transformer_output = self.transformer_encoder(sequence)
        game_token_output = transformer_output[:, 0, :]  # (1, dim)

        value = self.value_head(game_token_output)
        return game_token_output, value

    def predict_single(
        self, state_with_key: StateWithKey, legal_actions: List[ActionType]
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Performs a forward pass for a single state during MCTS search.
        Returns a dictionary of action priors and a value estimate.
        """
        self.eval()
        with torch.no_grad():
            state = self._apply_transforms(state_with_key.state)
            game_embedding, value_tensor = self._get_state_embedding_and_value(state)
            value = value_tensor.squeeze().cpu().item()

            if not legal_actions:
                return {}, value

            # Score legal actions
            action_tokens = self._actions_to_tokens(legal_actions)
            state_embedding_expanded = game_embedding.expand(len(legal_actions), -1)
            policy_input = torch.cat([state_embedding_expanded, action_tokens], dim=1)
            scores = self.policy_head(policy_input).squeeze(-1)
            policy_probs = F.softmax(scores, dim=0)

            policy_dict = {
                action: prob.item() for action, prob in zip(legal_actions, policy_probs)
            }

            return policy_dict, value

    def forward(
        self,
        state: Dict[str, "DataFrame"],
        legal_actions: List[List[ActionType]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for a batch of states during training.
        The input is a dictionary of batched DataFrames.
        `legal_actions` is a list of lists, where each inner list contains the
        legal actions for a state in the batch.
        """
        batch_size = len(legal_actions)
        transformer_output = self._get_transformer_output(
            state=state,
            batch_size=len(legal_actions),
        )
        game_token_output = transformer_output[:, 0, :]  # (batch, dim)
        device = transformer_output.device

        # --- Value Head ---
        value_preds = self.value_head(game_token_output).squeeze(-1)

        # --- Policy Head ---
        # This part becomes more complex because each item in the batch can have a
        # different number of legal actions. We process them all together and then
        # group the results.

        flat_action_tokens = []
        batch_indices_for_policy = []
        for i in range(batch_size):
            if not legal_actions[i]:
                continue
            action_tokens = self._actions_to_tokens(legal_actions[i])
            flat_action_tokens.append(action_tokens)
            batch_indices_for_policy.extend([i] * len(legal_actions[i]))

        if not batch_indices_for_policy:
            # If no legal actions in the entire batch, return empty logits.
            # This needs to be handled by the caller/loss function.
            return torch.empty(batch_size, 0, device=device), value_preds

        flat_action_tokens_tensor = torch.cat(flat_action_tokens, dim=0)
        batch_indices_tensor = torch.tensor(
            batch_indices_for_policy, device=device, dtype=torch.long
        )

        # Select the corresponding state embeddings for each action
        state_embs_for_policy = game_token_output[batch_indices_tensor]

        # Combine state and action embeddings and get scores
        policy_input = torch.cat(
            [state_embs_for_policy, flat_action_tokens_tensor], dim=1
        )
        scores = self.policy_head(policy_input).squeeze(-1)

        # The output `scores` is a flat tensor. The loss function will need to
        # handle this. We'll return the raw scores and let the loss function apply softmax.
        # However, CrossEntropyLoss expects logits in a (N, C) tensor.
        # We will pad the scores for each batch item.
        scores_by_item = []
        start_idx = 0
        for i in range(batch_size):
            num_actions = len(legal_actions[i])
            if num_actions > 0:
                scores_by_item.append(scores[start_idx : start_idx + num_actions])
                start_idx += num_actions
            else:
                scores_by_item.append(torch.empty(0, device=device))

        # Pad the list of score tensors to create a single (batch_size, max_actions) tensor.
        policy_logits = nn.utils.rnn.pad_sequence(
            scores_by_item, batch_first=True, padding_value=-torch.inf
        )

        return policy_logits, value_preds

    def init_zero(self):
        """
        Initializes the weights of the policy and value heads to zero.
        """
        nn.init.constant_(self.value_head[0].weight, 0)
        nn.init.constant_(self.value_head[0].bias, 0)
        nn.init.constant_(self.policy_head[-1].weight, 0)
        nn.init.constant_(self.policy_head[-1].bias, 0)
