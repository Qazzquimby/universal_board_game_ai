Oh but for stochasticity
you can get a range over hidden state, sure, but the legal actions would be totally different in different successor states.
Could do (hidden state distribution) -> (specific hidden state) -> predict legal action set
But that would be a lot of repeated work per simulation step, and with continuous action tokens it wouldn't repeat edges

This document outlines the necessary steps to implement a proper MuZero agent within the existing codebase.

The implementation will require three new core components, encapsulated within a new neural network architecture (`MuZeroNet`), and significant modifications to the MCTS algorithm.

- The state is represented by a dictionary of DataFrames, which are converted to tokens.
- The action space is dynamic and potentially large; we cannot assume a fixed superset of actions.
- Well its probably okay to repeat samples, maybe have 5 randomizations. "Progressive Widening"?
- https://aistudio.google.com/prompts/1rWCyLuGL2ruG2XYIEn0p9lHwNxjv6U97
Leave action generation as a stub for now


hidden states should actually be a mean and std so we can generate stochastic states.
We don't need reward at this time. Value is enough.

def get_hidden_state(state_tokens, game_token, action_tokens) -> hidden_state_vector:
    """Representation Function h"""
    tokens go through transformer encoder
    return (state tokens mean) concat (state tokens max) concat (game token)
Input is the raw game state and legal move list.
Can largely reuse embedding layers and transformer encoder in AlphaZeroNet.

def get_next_hidden_state(previous_hidden_state, action_token) -> hidden_state, value:
    """Dynamics function g""")
Can reuse embedding logic from AlphaZeroNet for the action?
Can probably just be an MLP for next hidden state and an MLP for value.

def get_hidden_state_action_tokens(hidden_state) ?
...

def get_policy_and_value(hidden_state) -> list scores corresponding to policy tokens, value:
Can reuse from AlphaZeroNet

Will need a MuZeroNet that shares most code with AZ but adds the new required heads


MCTS statewithkey will need to accept the hidden state vector
Make variants of the MCTS components such that:
- Root calls get hidden state
- env uses get next hidden state instead of env.step 
- expansion and evaluation use muzero's get_policy_and_value

Probably needs new `SelectionStrategy`, `ExpansionStrategy`, and `EvaluationStrategy` that interact with the `MuZeroNet`.

Need train_muzero.py and `agents/muzero_agent.py`, which should reuse from train_alphazero, or move shared code to a new file.

Data collection is same as alphazero

---
Unrolling loss calculation
For a fixed number of steps, eg 5

Get initial state to hidden and it's policy and value,
get loss for policy and loss for value

Given the taken action, predict the next hidden state and its own policy and value and take the loss 

At this time we won't use reanalyse. We'll use it later.

`MuZeroAgent` Implementation has the unrolling and loss calculation logic described above.
It will need to sample full trajectories from the replay buffer, not just individual steps.