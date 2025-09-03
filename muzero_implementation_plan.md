# MuZero Implementation Plan

## 1. Introduction

This document outlines the necessary steps to implement a proper MuZero agent within the existing codebase.

The implementation will require three new core components, encapsulated within a new neural network architecture (`MuZeroNet`), and significant modifications to the MCTS algorithm.

- The state is represented by a dictionary of DataFrames, which are converted to tokens.
- The action space is dynamic and potentially large; we cannot assume a fixed superset of actions.

## 2. Core MuZero Components

A MuZero model consists of three main functions, which we will implement in a single `MuZeroNet` module:

1.  **Representation Function (h):**
    -   **Input:** The raw game state (dictionary of DataFrames).
    -   **Output:** An initial *hidden state* (`s_0`). This is a fixed-size vector that summarizes the initial observation.
    -   **Implementation:** We can largely reuse the existing `AlphaZeroNet`'s embedding layers and transformer encoder. The hidden state `s_0` can be derived from the transformer's output, for example, by taking the embedding corresponding to the special `[GAME]` token.

2.  **Dynamics Function (g):**
    -   **Input:** A previous hidden state (`s_{k-1}`) and an action (`a_k`).
    -   **Output:** A predicted reward for taking that action (`r_k`) and the next hidden state (`s_k`).
    -   **Implementation:** This is a completely new component. A simple approach is to:
        -   Convert the action `a_k` into an embedding token (reusing the embedding logic from `AlphaZeroNet`).
        -   Concatenate the hidden state `s_{k-1}` and the action embedding.
        -   Pass this combined vector through an MLP to produce the next hidden state `s_k`.
        -   Add a separate MLP head (the "reward head") to the dynamics function to predict the scalar reward `r_k`.

3.  **Prediction Function (f):**
    -   **Input:** A hidden state (`s_k`).
    -   **Output:** A predicted policy (`p_k`) and value (`v_k`).
    -   **Implementation:** This is analogous to the policy and value heads in `AlphaZeroNet`. We can reuse the existing `policy_head` and `value_head`, which take the hidden state as input.

## 3. `MuZeroNet` Architecture (`models/networks.py`)

A new class `MuZeroNet` should be created.

```python
class MuZeroNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Reuse from AlphaZeroNet
        self.embedding_layers = ...
        self.transformer_encoder = ...
        self.policy_head = ...
        self.value_head = ...

        # New components
        self.dynamics_network = ... # E.g., an MLP
        self.reward_head = ... # E.g., a linear layer

    def representation(self, state: StateType) -> torch.Tensor:
        # state -> tokens -> transformer -> hidden_state
        # ...
        return hidden_state_s0

    def dynamics(self, hidden_state: torch.Tensor, action: ActionType) -> Tuple[torch.Tensor, torch.Tensor]:
        # action -> action_embedding
        # combined = concat(hidden_state, action_embedding)
        # next_hidden_state = self.dynamics_network(combined)
        # reward = self.reward_head(next_hidden_state) # or from `combined`
        # ...
        return next_hidden_state, reward

    def prediction(self, hidden_state: torch.Tensor, legal_actions: List[ActionType]) -> Tuple[Dict[ActionType, float], float]:
        # policy_logits = self.policy_head(...)
        # value = self.value_head(hidden_state)
        # ...
        return policy_dict, value

    # These methods will be called by the agent/MCTS
    def initial_inference(self, state: StateType, legal_actions: List[ActionType]) -> Tuple[torch.Tensor, Dict[ActionType, float], float]:
        hidden_state = self.representation(state)
        policy_dict, value = self.prediction(hidden_state, legal_actions)
        return hidden_state, policy_dict, value

    def recurrent_inference(self, hidden_state: torch.Tensor, action: ActionType, legal_actions: List[ActionType]) -> Tuple[torch.Tensor, float, Dict[ActionType, float], float]:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_dict, value = self.prediction(next_hidden_state, legal_actions)
        return next_hidden_state, reward, policy_dict, value
```

## 4. MCTS Algorithm Modifications (`algorithms/mcts.py`)

The core of the MCTS search must be updated to use the learned model instead of the environment.

1.  **`MCTSNode` State:**
    -   The `MCTSNode` will no longer store a `StateWithKey`. Instead, it will store the **hidden state** vector.
    -   It should also store the predicted reward for the action that led to this node.

2.  **Search Process:**
    -   **Root Node:** At the start of a search, the agent calls `MuZeroNet.initial_inference()` with the real environment state. This provides the root node's hidden state (`s_0`), policy priors, and value.
    -   **Selection:** When traversing the tree, instead of calling `env.step()`, the MCTS will:
        1.  Select an action `a_k` based on the UCB score (or PUCT).
        2.  Retrieve the next node from the current node's children. This child node already contains the next hidden state `s_k`.
    -   **Expansion & Evaluation:** When a leaf node is reached, instead of calling the environment, the agent will call `MuZeroNet.recurrent_inference()` with the leaf's hidden state and the selected action. This will:
        1.  Compute the next hidden state `s_k` and reward `r_k` via the dynamics function `g`.
        2.  Compute the policy `p_k` and value `v_k` for the new state via the prediction function `f`.
        3.  A new `MCTSNode` is created with `s_k` and expanded with the policy priors `p_k`. The edge leading to it stores the reward `r_k`.

3.  **Value and UCB Score:**
    -   The value of an edge, `Q(s, a)`, must now incorporate the predicted reward: `Q(s, a) = r(s,a) + gamma * V(s')`.
    -   The UCB score formula will use this updated `Q(s, a)`.

This requires creating new MuZero-specific implementations of `SelectionStrategy`, `ExpansionStrategy`, and `EvaluationStrategy` that interact with the `MuZeroNet`.

## 5. Training Process (`train.py` and `agents/muzero_agent.py`)

The training loop needs to be updated to compute MuZero's unique loss function over unrolled trajectories.

1.  **Data Collection:** Self-play remains the same. The agent uses the MuZero MCTS to play games, and we store trajectories of `(state, action, policy_target, reward)`. The `reward` will be 0 for all steps until the terminal state, where it is the game outcome.

2.  **Loss Calculation (Unrolling):**
    -   For each sample from the replay buffer, we perform a "training unroll" for a fixed number of steps (e.g., `K=5`).
    -   **Step 0 (Initial state):**
        -   `s_0 = h(state_0)`
        -   `p_0, v_0 = f(s_0)`
        -   Calculate loss: `L_0 = loss_value(v_0, z_0) + loss_policy(p_0, pi_0)` where `z_0` is the true discounted game outcome and `pi_0` is the MCTS policy target.
    -   **Step k=1 to K (Recurrent steps):**
        -   `r_k, s_k = g(s_{k-1}, a_k)` where `a_k` is the *actual action taken* in the game at that step.
        -   `p_k, v_k = f(s_k)`
        -   Calculate loss: `L_k = loss_value(v_k, z_k) + loss_policy(p_k, pi_k) + loss_reward(r_k, actual_reward_k)`.
    -   **Total Loss:** The final loss is the sum of the initial loss and the average of the recurrent losses, summed over the unroll steps. Gradient scaling might be needed for the recurrent steps.

3.  **`MuZeroAgent` Implementation:**
    -   This agent will contain the `MuZeroNet` and its optimizer.
    -   The `train_network` method will implement the unrolling and loss calculation logic described above.
    -   It will need to sample full trajectories from the replay buffer, not just individual steps.

## 6. Summary of Required Code Changes

-   **`models/networks.py`:**
    -   Create `MuZeroNet` with `representation`, `dynamics`, and `prediction` functions.
-   **`algorithms/mcts.py`:**
    -   Modify `MCTSNode` to store a hidden state and reward.
    -   Create new MCTS strategy classes (`MuZeroSelection`, `MuZeroExpansion`) that call the `MuZeroNet`'s recurrent inference instead of `env.step()`.
-   **`agents/muzero_agent.py`:**
    -   Implement `MuZeroAgent` to orchestrate the new network and MCTS.
    -   Implement the `train_network` method with the unrolling logic.
-   **`train.py`:**
    -   Update the data loading and training loop to support trajectory-based sampling and the new training logic in `MuZeroAgent`.
    -   Ensure game logs (`.json` files) store enough information (full trajectories) for MuZero's training needs. The current format seems to be step-by-step, which is fine, but the data loader will need to reconstruct trajectories.
