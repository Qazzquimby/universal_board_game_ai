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

2. **`MCTS` (Orchestrator):**
    * Responsibilities: Manage the search tree (root node), run the simulation loop, coordinate
      strategy objects.
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
        * `reset() -> None`: Resets the tree (new root node).
    * *Network Handling:* Strategies that require network predictions (e.g., `NetworkEvaluation`,
      `NetworkExpansion`) will need access to the network. This could be passed during
      initialization or requested via a callback/interface provided by the main `MCTS` object.

3. **Strategy Interfaces (Abstract Base Classes):**
    * `SelectionStrategy(ABC)`:
        * `select(node: MCTSNode, env: BaseEnvironment) -> Tuple[ActionType, MCTSNode]`: Choose the
          best action/child node to explore further based on a selection criterion (e.g., UCB1,
          PUCT).
    * `ExpansionStrategy(ABC)`:
        *
        `expand(node: MCTSNode, env: BaseEnvironment, network_output: Optional[Any] = None) -> None`:
        Add children to a leaf node based on legal actions. May use `network_output` (e.g., policy
        priors) if provided by the evaluation strategy.
    * `EvaluationStrategy(ABC)`:
        * `evaluate(node: MCTSNode, env: BaseEnvironment) -> Any`: Determine the value of a leaf
          node.
            * *Output:* Could be just a `float` (value) for rollouts, or a tuple
              `(value: float, policy_priors: Optional[np.ndarray])` if using a network, allowing the
              `ExpansionStrategy` to use the priors.
    * `BackpropagationStrategy(ABC)`:
        * `backpropagate(path: List[MCTSNode], value: float) -> None`: Update statistics (
          `visit_count`, `total_value`) of nodes along the simulation path, handling player
          perspective shifts.

4. **Concrete Strategy Implementations (Examples):**
    * Selection: `UCB1Selection(exploration_constant)`, `PUCTSelection(exploration_constant)`.
    * Expansion: `UniformExpansion`, `NetworkExpansion` (uses policy priors from `evaluate`). May
      handle Dirichlet noise.
    * Evaluation: `RandomRolloutEvaluation(discount_factor, max_depth)`,
      `NetworkEvaluation(network_interface)`.
    * Backpropagation: `StandardBackpropagation`.

5. **Configuration & Assembly:**
    * Use dataclasses (e.g., `MCTSConfig`, extending existing `AlphaZeroConfig` or creating new
      ones) to hold parameters.
    * Factory functions: `create_mcts(config, network=None)` that selects and initializes the
      correct strategies based on config flags (e.g., `config.use_network_evaluation`,
      `config.selection_policy='PUCT'`).

6. **Handling Variants & Features:**
    * **Pure MCTS:** Assemble with `UCB1Selection`, `UniformExpansion`, `RandomRolloutEvaluation`,
      `StandardBackpropagation`.
    * **AlphaZero:** Assemble with `PUCTSelection`, `NetworkExpansion`, `NetworkEvaluation`,
      `StandardBackpropagation`. Pass network interface.
    * **MuZero:** Plan as a future extension. Requires changes to Node, new strategies for
      dynamics/prediction.
    * **Early Stopping:** Implement checks within `MCTS.search` based on config flags (
      `dynamic_simulations_enabled`, etc.).
    * **Debugging:** Use `loguru`. Add `debug: bool` flag to config, passed to strategies to enable
      verbose logging or internal checks.

**Phased Implementation Plan:**

1. **Phase 1: Core Structure & Pure MCTS:**
    * Define `MCTSNode`.
    * Define Strategy ABCs.
    * Implement `MCTS` orchestrator structure.
    * Implement concrete strategies for Pure MCTS (`UCB1`, `UniformExpansion`, `RandomRollout`,
      `StandardBackprop`).
    * Refactor `MCTSAgent` to use this new structure.
    * Add basic tests.
2. **Phase 2: AlphaZero Integration:**
    * Implement concrete strategies for AlphaZero (`PUCT`, `NetworkExpansion`, `NetworkEvaluation`).
    * Integrate network handling.
    * Add Dirichlet noise option (likely in `NetworkExpansion` or `MCTS` orchestrator after root
      expansion).
    * Refactor `AlphaZeroMCTS` logic into this structure (potentially removing the old
      `AlphaZeroMCTS` class or making it a thin wrapper/factory).
    * Update `SelfPlayWorkerActor` to use the refactored MCTS.
    * Add tests for AlphaZero components.
3. **Phase 3: Features & Refinements:**
    * Implement robust early stopping (`dynamic_simulations`).
    * Enhance debugging capabilities.
    * Review efficiency and optimize critical paths.
4. **Phase 4: Advanced Variants (Future):**
    * MuZero adaptation.
    * Handling stochastic environments:
        * Requires modifications to handle probabilistic state transitions after actions.
        * Options include introducing explicit "chance nodes" in the tree or averaging outcomes within action nodes.
        * Impacts `SelectionStrategy`, `BackpropagationStrategy`, potentially `MCTSNode`, and how `EvaluationStrategy` (like rollouts) handles randomness.
        * The `BaseEnvironment` might need to expose outcome probabilities.
    * Handling hidden information (Partial Observability):
        * Requires searching over belief states or information sets rather than fully known states.
        * **Information Set MCTS (ISMCTS):** Build the tree over information sets. Nodes represent sets of possible states. Evaluation often involves sampling determinizations (concrete states consistent with the info set) and simulating from them. Requires significant changes to Node structure, state handling, and strategies.
        * **Determinization:** Sample one or more possible complete states (determinizations) consistent with current observations. Run a standard MCTS search for each determinization. Aggregate results. Less invasive to the core MCTS logic but requires management outside the main search function.
