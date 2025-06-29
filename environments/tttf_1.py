from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from environments.base import (
    BaseEnvironment,
    StateType,
    SanityCheckState,
    ActionResult,
    StateWithKey,
)
from engine.game_engine import GameEngine, GameEntity


@dataclass
class AttackAction:
    """Action to attack a target hero."""

    target_hero_id: int


ActionType = Union[AttackAction]


@dataclass
class DamageEvent:
    """Event representing damage dealt to a hero."""

    actor: "Hero"
    target: "Hero"
    amount: int

    def resolve(self):
        self.target.health -= self.amount


class Hero(GameEntity):
    """Represents a hero character in the game."""

    def __init__(self, game: GameEngine, name: str, player_owner_id: int, health: int):
        super().__init__(game=game, name=name)
        self.player_owner_id = player_owner_id
        self.health = health
        self.max_health = health

    def is_alive(self) -> bool:
        return self.health > 0

    def __repr__(self):
        return f"<Hero {self.name}|p{self.player_owner_id}|hp:{self.health}>"


class TTTF_1(BaseEnvironment):
    """
    A simple team-based tactical environment using the game engine.
    - Two players, each with N heroes.
    - Turn-based, alternating between players' heroes.
    - Last team with a hero standing wins.
    """

    metadata = {"name": "tttf_1"}

    def __init__(
        self,
        num_players: int = 2,
        num_heroes_per_player: int = 2,
        hero_health: int = 10,
    ):
        super().__init__()
        assert num_players == 2, "TTTF_1 currently only supports 2 players."
        self._num_players = num_players
        self._num_heroes_per_player = num_heroes_per_player
        self._hero_health = hero_health

        self.engine: Optional[GameEngine] = None
        self.heroes: List[Hero] = []
        self.turn_order_ids: List[int] = []
        self.current_turn_index: int = 0
        self.winner: Optional[int] = None

        self.reset()

        damage_action = self.engine.define_action("damage", DamageEvent)
        # could maybe be global instead if engine is global

    @property
    def num_players(self) -> int:
        return self._num_players

    # TODO later, replace this with one tensor size per entity type, including player, and one for Game
    # Will be input to transformer architecture
    @property
    def observation_tensor_shape(self) -> Tuple[int, ...]:
        # Vector: [h1_hp, h2_hp, ..., hn_hp]
        num_features = len(self.heroes)
        return (num_features,)

    # TODO later, replace this with one head per verb and choice type
    @property
    def policy_vector_size(self) -> int:
        # Max possible actions: attack any of the heroes.
        return len(self.heroes)

    def _reset(self) -> StateWithKey:
        self.engine = GameEngine()
        self.heroes = []

        # Create heroes in alternating turn order
        for i in range(self._num_heroes_per_player):
            for j in range(self._num_players):
                player_id = j
                hero_name = f"p{player_id}_h{i}"
                hero = Hero(self.engine, hero_name, player_id, self._hero_health)
                self.heroes.append(hero)

        self.turn_order_ids = list(range(len(self.heroes)))
        self.current_turn_index = 0
        self.winner = None
        self.done = False

        return self.get_state_with_key()

    def _step(self, action: ActionType) -> ActionResult:
        if self.done:
            return ActionResult(
                next_state_with_key=self.get_state_with_key(),
                reward=0.0,
                done=True,
            )

        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        acting_player_id = acting_hero.player_owner_id

        if isinstance(action, AttackAction):
            target_hero = self.heroes[action.target_hero_id]
            self.damage_action(actor=acting_hero, target=target_hero, amount=1)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        self.engine.run()
        self._check_for_winner()

        reward = 0.0
        if self.done:
            if self.winner == acting_player_id:
                reward = 1.0
            elif self.winner is not None:
                reward = -1.0
        else:
            self._advance_turn()

        return ActionResult(
            next_state_with_key=self.get_state_with_key(),
            reward=reward,
            done=self.done,
        )

    def _advance_turn(self):
        num_heroes = len(self.heroes)
        for i in range(1, num_heroes + 1):
            next_index = (self.current_turn_index + i) % num_heroes
            acting_hero_id = self.turn_order_ids[next_index]
            if self.heroes[acting_hero_id].is_alive():
                self.current_turn_index = next_index
                return

        # All heroes are dead
        self.done = True

    def _check_for_winner(self):
        player_heroes_alive = [0] * self._num_players
        for hero in self.heroes:
            if hero.is_alive():
                player_heroes_alive[hero.player_owner_id] += 1

        alive_players = [p for p, count in enumerate(player_heroes_alive) if count > 0]

        if len(alive_players) <= 1:
            self.done = True
            self.winner = alive_players[0] if alive_players else None

    def get_legal_actions(self) -> List[ActionType]:
        if self.done:
            return []

        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        acting_player_id = acting_hero.player_owner_id

        legal_actions = []
        for i, hero in enumerate(self.heroes):
            if hero.is_alive() and hero.player_owner_id != acting_player_id:
                legal_actions.append(AttackAction(target_hero_id=i))

        return legal_actions

    def get_current_player(self) -> int:
        # Before a move, current_turn_index points to the currently acting hero.
        acting_hero_id = self.turn_order_ids[self.current_turn_index]
        return self.heroes[acting_hero_id].player_owner_id

    def _get_state(self) -> StateType:
        return {
            "hero_healths": [h.health for h in self.heroes],
            "current_turn_index": self.current_turn_index,
            "done": self.done,
            "winner": self.winner,
        }

    def get_winning_player(self) -> Optional[int]:
        return self.winner

    def copy(self) -> "TTTF_1":
        new_env = TTTF_1(
            num_players=self._num_players,
            num_heroes_per_player=self._num_heroes_per_player,
            hero_health=self._hero_health,
        )
        new_env.set_state(self._get_state())
        return new_env

    def set_state(self, state: StateType) -> None:
        # To restore state, we need to rebuild the hero list and engine state.
        # A full reset is easiest to ensure engine is clean.
        self._reset()

        # Apply the saved state
        healths = state["hero_healths"]
        for i, hero in enumerate(self.heroes):
            hero.health = healths[i]

        self.current_turn_index = state["current_turn_index"]
        self.done = state["done"]
        self.winner = state["winner"]

        self._dirty = True

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        states = []

        # State 1: Initial state
        initial_env = TTTF_1(
            num_players=self._num_players,
            num_heroes_per_player=self._num_heroes_per_player,
            hero_health=self._hero_health,
        )
        states.append(
            SanityCheckState(
                description="Initial state",
                state_with_key=initial_env.get_state_with_key(),
                expected_value=0.0,
                expected_action=None,
            )
        )

        # State 2: Player 0 can win
        if self._num_players == 2 and self._num_heroes_per_player >= 1:
            win_env = TTTF_1(
                num_players=2,
                num_heroes_per_player=self._num_heroes_per_player,
                hero_health=10,
            )
            # P0 is player 0 (heroes with even IDs), P1 is player 1 (odd IDs)
            # P0's turn (hero 0), P1 has one hero left with 1 HP.
            # Kill all of P1's heroes except one.
            for i, hero in enumerate(win_env.heroes):
                if hero.player_owner_id == 1:
                    hero.health = 0
            # Set one P1 hero to 1 health
            win_env.heroes[1].health = 1

            win_env.current_turn_index = 0  # P0's turn, hero 0 acts
            win_env._dirty = True

            winning_action = AttackAction(target_hero_id=1)
            states.append(
                SanityCheckState(
                    description="Player 0 can win by attacking hero 1",
                    state_with_key=win_env.get_state_with_key(),
                    expected_value=1.0,
                    expected_action=winning_action,
                )
            )

        return states

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print("-" * 20)
            if self.done:
                print(f"Game Over! Winner: Player {self.winner}")
            else:
                acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
                print(
                    f"Turn: {self.current_turn_index}, "
                    f"Player {self.get_current_player()}'s turn ({acting_hero.name} acts)"
                )

            print("Heroes:")
            for i, hero in enumerate(self.heroes):
                status = "Alive" if hero.is_alive() else "Dead"
                print(
                    f"  ID {i}: {hero.name} (P{hero.player_owner_id}) - "
                    f"HP: {hero.health}/{hero.max_health} [{status}]"
                )
            print("-" * 20)
