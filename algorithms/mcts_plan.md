# MCTS Refactoring Plan

**Goal:** Create a flexible, extensible MCTS framework allowing easy implementation and combination
of different MCTS variants (Pure MCTS, AlphaZero, MuZero-like components) and features (debugging,
early stopping, stochasticity, hidden info).

**Core Idea:** Decouple the main MCTS search loop from the specific algorithms used in each phase (
Selection, Expansion, Evaluation, Backpropagation) using a Strategy pattern.

**Proposed Structure:**

1. **`MCTSNode`:**
    * Responsibilities: Store node-specific data.
    * Attributes: `parent: Optional[MCTSNode]`, `children: Dict[ActionType, MCTSNode]`,
      `visit_count: int`, `total_value: float`, `prior: float` (optional, for network-based priors).
    * Methods: `value() -> float` (Calculates Q-value: `total_value / visit_count`),
      `is_expanded() -> bool`.
    * *Future (MuZero):* May need to store a hidden state representation.

2. **`MCTSOrchestrator` (Core Class):**
    * Responsibilities: Manage the search tree (root node), run the main simulation loop, coordinate
      strategy objects. This class should eventually be the *only* MCTS runner, configured via
      strategies.
    * Initialization: Takes strategy objects (`SelectionStrategy`, `ExpansionStrategy`,
      `EvaluationStrategy`, `BackpropagationStrategy`), configuration (`num_simulations`,
      `discount_factor`, `cpuct` etc.), optionally a network interface.
    * Methods:
        * `search(env: BaseEnvironment, state: StateType) -> MCTSNode`: Runs the main MCTS loop,
          returns the root node after search.
        * `_run_one_simulation(env: BaseEnvironment)`: Performs one iteration (Select -> Expand ->
          Evaluate -> Backpropagate). Needs careful handling of environment state copies.
        *
      `get_policy(temperature: float = 1.0) -> Tuple[ActionType, Dict[ActionType, float], Dict[ActionType, int]]`:
      Calculates the final action policy based on root children visits (returns chosen action,
      probabilities, visits).
        * `advance_root(action: ActionType) -> None`: Moves the root down the tree for reuse.
        * `reset_root() -> None`: Resets the tree (new root node).
    * *Simulation State:* The `search` method typically creates a fresh copy of the environment
      state for *each* simulation run, starting from the root state. This is simpler but doesn't
      reuse computation *within* a single `search` call beyond the tree structure itself.
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
      Select a path from the starting node down to a leaf node. Returns the `path` (list of nodes
      from start to leaf, inclusive), the `leaf_node`, and the `leaf_env` state corresponding to
      the leaf node.
    * `ExpansionStrategy(ABC)`:
        *
      `expand(node: MCTSNode, env: BaseEnvironment, network_output: Optional[Any] = None) -> None`:
      Add children to a leaf node based on legal actions. May use `network_output` (e.g., policy
      priors from `EvaluationStrategy`).
    * `EvaluationStrategy(ABC)`:
        * `evaluate(node: MCTSNode, env: BaseEnvironment) -> float`: Determine the value of a leaf
          node.
    * `BackpropagationStrategy(ABC)`:
        * `backpropagate(path: List[MCTSNode], value: float, player_at_leaf: int) -> None`: Update
          statistics (`visit_count`, `total_value`) of nodes along the simulation path. Should work
          for any number of players.

4. **Concrete Strategy Implementations (Examples):**
    * Selection: `UCB1Selection(exploration_constant)`, `PUCTSelection(exploration_constant)`.
    * Expansion: `UniformExpansion`, `NetworkExpansion` (uses policy priors from `evaluate`). May
      handle Dirichlet noise.
    * Evaluation: `RandomRolloutEvaluation(discount_factor, max_depth)`,
      `NetworkEvaluation(network_interface)`.
    * Backpropagation: `StandardBackpropagation`.

5. **Configuration & Assembly:**
    * Configure strategies with init params.
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

Early stopping should be a strategy. There are diverse stopping strategies I'd like tested.

Apply dirichlet noise could probably just be a toggle in search at this point.

  * **Debugging:** Use `loguru`. Add `debug: bool` flag to config, potentially passed to
    strategies for verbose logging.
