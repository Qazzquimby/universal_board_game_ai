# MCTS Refactoring Plan

**Goal:** Create a flexible, extensible MCTS framework allowing easy implementation and combination
of different MCTS variants (Pure MCTS, AlphaZero, MuZero-like components) and features (debugging,
early stopping, stochasticity, hidden info).

**Core Idea:** Decouple the main MCTS search loop from the specific algorithms used in each phase (
Selection, Expansion, Evaluation, Backpropagation) using a Strategy pattern.

**Proposed Structure:**

1. **`MCTSNode`:**
    * Responsibilities: Store node-specific data. Represents a game state (perfect info) or an
      information set (hidden info).
    * Attributes: `parent: Optional[MCTSNode]`, `children: Dict[ActionType, MCTSNode]`,
      `visit_count: int`, `total_value: float`, `prior: float` (optional, for network-based priors).
      May also store the `information_set` or `state` it represents.
    * Methods: `value() -> float` (Calculates Q-value: `total_value / visit_count`),
      `is_expanded() -> bool`.
    * *Stochastic Handling:* May need `node_type: str` ('decision' or 'chance') and potentially
      `outcome_probabilities: Dict[StateType, float]` if it's a chance node. Children of decision
      nodes are keyed by `ActionType`. Children of chance nodes represent possible outcomes (leading
      to new decision nodes) and might be stored differently (e.g., list of `(prob, node)` tuples).
    * *Hidden Information:* When representing an information set, children correspond to actions,
      leading to the *next* information set for the acting player.
    * *Future (MuZero):* May need to store a hidden state representation.

2. **`MCTSOrchestrator` (Core Class):**
    * Responsibilities: Manage the search tree (root node), run the main simulation loop, coordinate
      strategy objects. This class should eventually be the *only* MCTS runner, configured via
      strategies. Handles decision/chance nodes and potentially information sets/determinization.
    * Initialization: Takes strategy objects (`SelectionStrategy`, `ExpansionStrategy`,
      `EvaluationStrategy`, `BackpropagationStrategy`), configuration (`num_simulations`,
      `discount_factor`, `cpuct` etc.), optionally a network interface.
    * Methods:
        * `search(env: BaseEnvironment, root_info_set: StateType) -> MCTSNode`: Runs the main MCTS loop
          starting from the player's current `root_info_set`. Returns the root node after search.
        * `_run_one_simulation(env: BaseEnvironment, root_info_set: StateType)`: Performs one iteration.
          * *Hidden Info (IS-MCTS):* Sample a determinization from `root_info_set`. Run
            Select -> [Handle Chance] -> Expand -> Evaluate -> Backpropagate on the *determinized* state.
          * *Perfect Info:* Run Select -> [Handle Chance] -> Expand -> Evaluate -> Backpropagate directly.
          Needs careful handling of environment state copies, transitions through chance nodes, and
          mapping results back to the information set tree.
        *
      `get_policy(temperature: float = 1.0) -> Tuple[ActionType, Dict[ActionType, float], Dict[ActionType, int]]`:
      Calculates the final action policy based on root children visits (returns chosen action,
      probabilities, visits). Operates on the root node (representing an info set or state).
        * `advance_root(action: ActionType, next_info_set: Optional[StateType] = None) -> None`:
          Moves the root down the tree based on the taken `action`.
          * *Hidden Info:* The `next_info_set` observed after taking the action becomes the new root.
          * *Stochastic:* May involve intermediate chance nodes as before.
        * `reset_root() -> None`: Resets the tree (new root node).
    * *Simulation State:* For IS-MCTS, each simulation starts by sampling a determinization from the
      root information set. The simulation runs on this determinized state, but the results (visits, values)
      are aggregated back onto the main information set tree.
    * *Network Handling:* Strategies requiring network access (e.g., `NetworkEvaluation`,
      `NetworkExpansion`) should receive a `NetworkInterface` object during their initialization.
    * *Asynchronous Evaluation:* The core `MCTSOrchestrator` and its strategies should remain
      synchronous. Asynchronous operations (like batched network calls) should be handled by the
      *calling code* (e.g., an actor or training loop) which interacts with the MCTS orchestrator
      and the inference service.

3. **Strategy Interfaces (Abstract Base Classes):**
    * `SelectionStrategy(ABC)`:
        *
      `select(node: MCTSNode, env: BaseEnvironment) -> Tuple[List[MCTSNode], MCTSNode, BaseEnvironment]`:
      Select a path from the starting node down to a leaf node within the current simulation's
      context (which might be a determinized state in IS-MCTS). Must handle traversing chance nodes.
      Returns the `path` (list of nodes from start to leaf, inclusive), the `leaf_node`, and the
      `leaf_env` state corresponding to the leaf node *in the current simulation*.
    * `ExpansionStrategy(ABC)`:
        *
      `expand(node: MCTSNode, env: BaseEnvironment, network_output: Optional[Any] = None) -> None`:
      Expand a leaf node based on the legal actions available in the `env` state (which might be
      determinized). Creates children (representing next states or info sets) in the main tree.
      Handles stochastic transitions by potentially creating chance nodes. May use `network_output`.
    * `EvaluationStrategy(ABC)`:
        * `evaluate(node: MCTSNode, env: BaseEnvironment) -> float`: Determine the value of a leaf
          node, typically by simulating from the `env` state (which might be determinized) or using
          a network.
    * `BackpropagationStrategy(ABC)`:
        * `backpropagate(path: List[MCTSNode], value: float, player_at_leaf: int) -> None`: Update
          statistics (`visit_count`, `total_value`) of nodes along the simulation path in the main
          tree. Must handle decision/chance nodes. The `value` is derived from the simulation's outcome.

4. **Concrete Strategy Implementations (Examples):**
    * Selection: `UCB1Selection(exploration_constant)`, `PUCTSelection(exploration_constant)`.
    * Expansion: `UniformExpansion`, `NetworkExpansion` (uses policy priors from `evaluate`). May
      handle Dirichlet noise.
    * Evaluation: `RandomRolloutEvaluation(discount_factor, max_depth)` (runs rollout on determinized state),
      `NetworkEvaluation(network_interface)` (evaluates determinized state or info set representation).
    * Backpropagation: `StandardBackpropagation` (aggregates results onto the info set tree).

5. **Configuration & Assembly:**
    * Configure strategies with init params. May need flags like `is_hidden_info`.
    * Factory functions: `create_mcts_orchestrator(config, network=None)` that selects and
      initializes the `MCTSOrchestrator` with the correct strategies based on config flags (e.g.,
      `config.use_network_evaluation`, `config.selection_policy='PUCT'`).

6. **Handling Variants & Features:**
    * **Pure MCTS:** Assemble `MCTSOrchestrator` with `UCB1Selection`, `UniformExpansion`,
      `RandomRolloutEvaluation`, `StandardBackpropagation`.
    * **AlphaZero:** Assemble `MCTSOrchestrator` with `PUCTSelection`, `NetworkExpansion`,
      `NetworkEvaluation`, `StandardBackpropagation`. Pass network interface to relevant strategies.
    * **MuZero:** Plan as a future extension. Requires changes to Node, new strategies for
      dynamics/prediction.
    * **Stochastic Games:** Assemble `MCTSOrchestrator` with strategies aware of decision/chance
      nodes. Requires environment support for identifying stochastic transitions and
      outcomes/probabilities.
    * **Hidden Information Games (IS-MCTS):** Assemble `MCTSOrchestrator`. The `_run_one_simulation`
      method handles determinization sampling. Strategies operate on the determinized state but
      update the main information set tree. Requires significant environment support for player
      observations and *efficient* determinization sampling (can be a bottleneck).

Early stopping should be a strategy. There are diverse stopping strategies I'd like tested.

Apply dirichlet noise could probably just be a toggle in search at this point.

* **Debugging:** Use `loguru`. Add `debug: bool` flag to config, potentially passed to
  strategies for verbose logging.
* **Environment Interaction:** Note the critical importance of `BaseEnvironment` implementation:
    * Stochastic: Identifying chance events, providing outcome probabilities/sampling.
    * Hidden Info: Providing player-specific observations (`get_observation(player_id)`),
      sampling determinizations (`sample_determinization(player_id, info_set)`), potentially
      returning updated info sets from `step`.
    * *Performance:* For complex games (MTG), methods like `copy`, `step`, `get_legal_actions`,
      and `sample_determinization` must be highly optimized. A slow environment will bottleneck MCTS.
    * *State Representation:* `StateType` needs careful design for simulation efficiency and network input.
* **Memory Management:** For deep searches or large trees (common in complex games), consider
  strategies for tree pruning, node recycling, or memory-efficient node representations.




Variants to support with strategies:
- alphazero network use
- muzero network use
- stochasticity handling
- hidden info handling
- progressive widening
- early stopping strategies



Is tree reuse a strategy or is it virtually always beneficial?
