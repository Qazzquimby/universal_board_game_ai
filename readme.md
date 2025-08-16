On muzero difficulties
https://aistudio.google.com/prompts/1trfIx7qtUauQZ-9WgecTgO6bP5RbP016



The goal is to create a universal board game engine which makes it easy (as it can be) to define any
board game, up to and including extremely complex board games like MtG.
The engine then automatically designs, parameterizes, and trains a powerful AI model to play the
game.


- Add wandb tracking. Want to keep current setup and allow slotting in new changes
- replace network for connect4 with resnet and compare



The current focus is on validating the AI generation part of the process.
The current unvalidated plan is to

- represent the gamestate as a heterogeneous graph
- use stochastic muzero with graph transformers
- pass the legal actions in as part of the game state rather than having a fixed policy network

# Validation roadmap

- [x] create evaluation environment and reusable base classes.
- [ ] implement alphazero to fair performance
  - [ ] Create tictactoe environment with decreased mcts iterations for faster evaluation
  - [ ] Move ml to gpu 
  - [ ] Batch leaves for ml processing
  - [ ] Run games in parallel via multithreading
- 
- 
- [ ] implement muzero
- [ ] create graph versions of environments and zero models and iterate until performance is okay
- [ ] instead of using fixed policy output, pass in legal actions as part of the gamestate and have
  variable sized policy output. Iterate until performance is okay.
- [ ] gradually work with more complex games. Add components needing embeddings, stochasticity, partial info, etc.


# parallelization plan

MCTS class has interface
- get_network_request
  - simulates until it reaches the next required network request
  - returns None if the game is resolved and no more requests are needed


```
Setup:

You have your main process.

You spawn multiple Worker processes using Python's multiprocessing module (e.g., N workers, where N is maybe the number of CPU cores minus one). Each worker gets a copy of the game environment and the current network weights.

You often have a dedicated GPU Actor process (or thread, if managed carefully) responsible only for running network inferences.

You use multiprocessing.Queues for communication: one queue for workers to send evaluation requests to the GPU actor, and potentially separate queues for the GPU actor to send results back to the specific workers.

Worker Process Loop (Simplified):

Start a new game.

While the game isn't over:

Begin MCTS for the current player/state.

Run simulations. When a simulation reaches a leaf node needing evaluation:

Package the game state (or MuZero hidden state) representation.

Put a request (worker_id, state_data) onto the inference_request_queue.

Pause this simulation path temporarily (or store its state).

Move to a different game, and again simulate until it needs a network request.

Periodically check its result_queue for completed evaluations sent back by the GPU actor.

When a result (policy, value) arrives for a specific pending node, update that node in the MCTS tree and continue the simulation(s) downwards from there.

After enough simulations, choose a move based on MCTS statistics, update the game state, store the (state, policy_target, value_target) data point.

When the game ends, send the collected trajectory data back to the main process (e.g., via another queue) for storage in the replay buffer.

GPU Actor Loop (Simplified):

Continuously monitor the inference_request_queue.

Collect requests until either a timeout occurs or a desired batch_size is reached (e.g., 64, 128, 256 requests).

Stack all the state_data from the collected requests into a single batch tensor.

Perform one forward pass of the network on the entire batch.

Un-batch the results (policies and values).

For each result, send it back to the originating worker using their worker_id (e.g., put (worker_id, policy, value) onto worker_results_queue[worker_id]).
```



# Ideas

Normally old self play data is discarded as low quality because it came from a previous version of
the model. Imagine if instead the model is periodically given an objective score and attach that
score to each move. Now rather than old play data being misleading because a move might lead to a
win despite being terrible, it'd show that the move lead to a win in a low score game. If all that
shows is "ignore this one" then it's pointless and you should just discard old data. If instead it's
able to get a better sense of what moves are good or poor based on the score of the player, then
it's a useful thing to do with masses of old data. In actual play the model would be trying to
output moves that look very high score.
We could save model score information during checkpoint evaluations and make a model variant that
uses it.
For objective scoring, could start with a score based on win rate against random agent, and then
later evaluate against the current strongest scored model.
My prediction is that low skill training data is objectively unhelpful, rather than something you
can use as negative examples, in the same way a random agent provides no information. This makes the
idea pointless.
