# DEBUG_MODE = False
# MAX_TRAINING_ROWS = 100_000
# MAX_GAMES = MAX_TRAINING_ROWS // 8 # guessing games will have at least 8 moves
#
# class ModelLayer(nn.Module, abc.ABC):
#     def __init__(self, in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
#         super().__init__()
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#
#     @property
#     def in_size(self):
#         return math.prod(self.in_shape)
#
#     @property
#     def out_size(self):
#         return math.prod(self.out_shape)
#
#     def forward(self, input):
#         return self._forward(input)
#
#     @abc.abstractmethod
#     def _forward(self, input):
#         """Subclasses implement their specific forward logic"""
#         raise NotImplementedError
#
#
# class Conv2d(ModelLayer):
#     def __init__(
#         self,
#         width: int,
#         height: int,
#         kernel_size: int,
#         in_channels: int,
#         out_channels: int,
#     ):
#         in_shape = (in_channels, height, width)
#         out_shape = (out_channels, height, width)
#         super().__init__(in_shape, out_shape)
#
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             padding="same",
#         )
#         self.activation = nn.ReLU()
#
#     def _forward(self, input):
#         return self.activation(self.conv(input))
#
#
# class Linear(ModelLayer):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__(in_shape=(-1, in_features), out_shape=(-1, out_features))
#         self.linear = nn.Linear(in_features, out_features)
#         self.activation = nn.ReLU()
#
#     def _forward(self, input):
#         return self.activation(self.linear(input))
#
#
# class TrainingRow:
#     def __init__(self, state: list[torch.Tensor], outcome: float, policy: torch.Tensor):
#         # Store original tensors with shapes for debugging
#         self.original_state = state
#
#         # Create flat version for model input
#         self.flat_state = torch.cat([t.flatten() for t in state]).float()
#         self.outcome: float = outcome
#         self.policy = policy.float()
#
#     def __iter__(self):
#         return iter((self.flat_state, self.outcome, self.policy))
#
#     @classmethod
#     def from_history_item(cls, history_item, game: "Game", zone_names=None):
#         state = history_item["state"]
#         state_as_list = []
#         if not zone_names:
#             zone_names = sorted(state.keys())
#
#         for zone_name in zone_names:
#             if zone_name.startswith("_") or zone_name == "outcome":
#                 continue
#             zone_state = state[zone_name]
#             if (
#                     isinstance(zone_state, dict)
#                     and "data" in zone_state
#                     and "shape" in zone_state
#             ):
#                 state_as_list.append(
#                     torch.tensor(zone_state["data"]).reshape(
#                         zone_state["shape"]).float()
#                 )
#             else:
#                 state_as_list.append(zone_state)
#
#         outcome = torch.tensor([state["outcome"]]).float()
#         policy_target = torch.zeros(game.get_policy_size())
#
#         if history_item["action"] is not None:
#             action_info = history_item["action"]
#             action_type = action_info["type"]
#             action_payload = action_info.get("payload", {})
#
#             # Find matching action class
#             action_cls = next(
#                 (cls for cls in game.action_types if
#                  cls.__name__.lower() == action_type),
#                 None,
#             )
#
#             if action_cls:
#                 # Calculate base index for this action type
#                 base_idx = sum(
#                     cls.get_action_space_size()
#                     for cls in game.action_types
#                     if cls != action_cls
#                 )
#
#                 # Calculate index within action type
#                 index = 0
#                 multiplier = 1
#                 params = reversed(list(action_cls._param_constraints.items()))
#
#                 for param_name, constraints in params:
#                     value = action_payload.get(param_name)
#                     min_val = constraints["min"]
#                     max_val = constraints["max"]
#                     index += (value - min_val) * multiplier
#                     multiplier *= max_val - min_val + 1
#
#                 policy_target[base_idx + index] = 1.0
#
#         return cls(
#             state=state_as_list,
#             outcome=outcome.item(),
#             policy=policy_target,
#         )
#
#
# class GameDataset(Dataset):
#     def __init__(self, game: "Game", num_games=MAX_GAMES, play_log_paths=None):
#         self.game = game
#         self.log_dir = PROJECT_ROOT / "play_logs" / game.name
#
#         if play_log_paths is None:
#             all_files = sorted(self.log_dir.glob("game_*.json"), key=os.path.getmtime)
#             self.play_log_paths = all_files[-num_games:] if num_games > 0 else []
#         else:
#             self.play_log_paths = play_log_paths
#
#         if DEBUG_MODE:
#             self.play_log_paths = self.play_log_paths[:100]
#
#         self.training_rows: Deque[TrainingRow] = deque(maxlen=MAX_TRAINING_ROWS)
#         self.zone_names = None
#         self.zone_shapes = []
#         self.load_data()
#
#     def load_data(self):
#         print("num game files", len(self.play_log_paths))
#
#         for play_log_path in self.play_log_paths:
#             with open(play_log_path, "r") as f:
#                 game_data = json.load(f)
#
#             if not self.zone_names or not self.zone_shapes:
#                 try:
#                     first_state = game_data["history"][0]["state"]
#                     self.zone_names = sorted(key for key in first_state.keys())
#                     self.zone_shapes = [
#                         tuple(zone["shape"])
#                         for zone in first_state.values()
#                         if isinstance(zone, dict) and "shape" in zone
#                     ]
#                 except (KeyError, IndexError):
#                     pass
#
#             for history_item in game_data["history"]:
#                 state_outcome_policy = TrainingRow.from_history_item(
#                     history_item, self.game, self.zone_names
#                 )
#                 self.training_rows.append(state_outcome_policy)
#
#         print("Data length", len(self.training_rows))
#
#     def __len__(self):
#         return len(self.training_rows)
#
#     def __getitem__(self, idx):
#         training_row = self.training_rows[idx]
#         return training_row
#
#     @staticmethod
#     def collate_fn(batch):
#         """Handle batching of StateOutcomePolicy items"""
#         states = torch.stack([row.flat_state for row in batch])
#         outcomes = torch.tensor([sop.outcome for sop in batch], dtype=torch.float32).unsqueeze(-1)
#         policies = torch.stack([sop.policy for sop in batch])
#         return states, outcomes, policies
#
# @dataclass
# class ModelResult:
#     value: float
#     policy: torch.Tensor
#
#
# class GameModel(pl.LightningModule):
#     def __init__(self, game: "Game"):
#         super().__init__()
#         self.game = game
#
#         self.input_size = sum(math.prod(shape) for shape in game.get_zone_shapes())
#
#         self.shared_net = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         )
#
#         self.value_head = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Tanh()
#         )
#
#         self.policy_head = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, game.get_policy_size()),
#             nn.Softmax(dim=-1)
#         )
#
#         self.value_loss = nn.MSELoss()
#         self.policy_loss = nn.CrossEntropyLoss()
#
#     def run(self, state: "State"):
#         # Convert state to model input format
#         state_tensors = state.get_tensors()
#
#         # Create properly formatted input tensor
#         flat_state = torch.cat(state_tensors).unsqueeze(0)  # Add batch dimension
#         flat_state = flat_state.to(self.device)
#
#         with torch.no_grad():
#             value_pred, policy_pred = self(flat_state)
#             value = value_pred.item()
#             policy_logits = policy_pred.squeeze().cpu().numpy()
#             return ModelResult(value=value, policy=policy_logits)
#
#     def forward(self, x):
#         x = self.shared_net(x)
#         return self.value_head(x), self.policy_head(x)
#
#     def training_step(self, batch, batch_idx):
#         x, value_target, policy_target = batch
#         value_pred, policy_pred = self(x)
#
#         val_loss = self.value_loss(value_pred, value_target)
#         pol_loss = self.policy_loss(policy_pred, policy_target)
#         total_loss = val_loss + pol_loss
#
#         self.log_dict({
#             "train_loss": total_loss,
#             "train_val_loss": val_loss,
#             "train_pol_loss": pol_loss
#         }, prog_bar=True)
#         return total_loss
#
#     def validation_step(self, batch, batch_idx):
#         x, value_target, policy_target = batch
#         value_pred, policy_pred = self(x)
#
#         val_loss = self.value_loss(value_pred, value_target)
#         pol_loss = self.policy_loss(policy_pred, policy_target)
#         total_loss = val_loss + pol_loss
#
#         self.log_dict({
#             "val_loss": total_loss,
#             "val_val_loss": val_loss,
#             "val_pol_loss": pol_loss
#         }, prog_bar=True)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=100, eta_min=1e-5
#         )
#         return [optimizer], [scheduler]
