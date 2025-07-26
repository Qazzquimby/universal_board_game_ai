# TODO move to own file, its not detached
class ActionsAreTokensNet(nn.Module):
    def __init__(self, state_model_params):
        super().__init__()
        embedding_dim = state_model_params.get("embedding_dim", 128)
        num_heads = state_model_params.get("num_heads", 8)

        self.state_model = StateModel(**state_model_params)

        self.policy_decoder = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
            dim_feedforward=embedding_dim * 4,
        )
        self.policy_head = nn.Linear(embedding_dim, 1)

    def forward(
        self, owner, coords, src_key_padding_mask, legal_moves, legal_moves_mask
    ):
        # legal_moves: (batch, max_num_legal_moves) tensor of column indices
        # legal_moves_mask: (batch, max_num_legal_moves) boolean tensor, True for padding

        state_tokens, value = self.state_model(owner, coords, src_key_padding_mask)
        # state_tokens (memory): (batch, num_pieces+1, embed_dim)

        # Embed legal moves. Reusing col_embedding from state_model.
        action_emb = self.state_model.col_embedding(
            legal_moves
        )  # (batch, max_num_legal_moves, embed_dim)

        batch_size = owner.shape[0]
        game_mask = torch.zeros(batch_size, 1, device=owner.device, dtype=torch.bool)
        memory_key_padding_mask = torch.cat((src_key_padding_mask, game_mask), dim=1)

        decoded_actions = self.policy_decoder(
            tgt=action_emb,
            memory=state_tokens,
            tgt_key_padding_mask=legal_moves_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        action_scores = self.policy_head(
            decoded_actions
        )  # (batch, max_num_legal_moves, 1)

        policy_output = torch.full(
            (batch_size, BOARD_WIDTH), -float("inf"), device=owner.device
        )

        batch_indices = (
            torch.arange(batch_size, device=owner.device)
            .unsqueeze(1)
            .expand_as(legal_moves)
        )

        # We only fill in values for non-padded legal moves
        valid_action_mask = ~legal_moves_mask

        policy_output[
            batch_indices[valid_action_mask], legal_moves[valid_action_mask]
        ] = action_scores[valid_action_mask].squeeze(-1)

        return policy_output, value
