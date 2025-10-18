import sys
from typing import Dict, Optional, Tuple

from loguru import logger

from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from core.agent_interface import Agent
from core.config import AppConfig
from environments.base import BaseEnvironment
from factories import get_agents, get_environment


def predict(
    agent: Agent, env: BaseEnvironment, network_only: bool = False
) -> Tuple[Optional[Dict], Optional[float]]:
    """
    Get policy and value prediction from an agent for a given environment state.
    """
    legal_actions = env.get_legal_actions()

    if isinstance(agent, AlphaZeroAgent) and network_only:
        if not agent.network:
            logger.warning("Agent has no network, cannot predict.")
            return None, None
        agent.network.eval()
        policy_dict, value = agent.network.predict_single(
            env.get_state_with_key(), legal_actions
        )
        return policy_dict, value

    elif isinstance(agent, (MCTSAgent, AlphaZeroAgent)):
        # Run MCTS search to populate the root node
        _ = agent.act(env, train=False)
        policy_result = agent.get_policy_from_visits(temperature=0.0)
        action_visits = policy_result.action_visits
        total_visits = sum(action_visits.values())

        if total_visits > 0:
            policy_dict = {
                action: visits / total_visits
                for action, visits in action_visits.items()
            }
        else:
            policy_dict = {}

        # Per user, only AZ network-only prediction should have a value.
        return policy_dict, None

    else:
        logger.warning(
            f"Prediction not implemented for agent type {type(agent).__name__}"
        )
        return None, None


def run_sanity_checks_for_agent(
    env: BaseEnvironment, agent: Agent, agent_name: str, network_only: bool = False
):
    """Runs predictions on predefined sanity check states for a given agent."""
    logger.info(f"\n--- Running Sanity Checks for Agent: '{agent_name}' ---")
    sanity_states = env.get_sanity_check_states()

    if not sanity_states:
        logger.info("No sanity check states defined for this environment.")
        return

    for check_case in sanity_states:
        logger.info(f"\nChecking State: {check_case.description}")
        temp_env = env.copy()
        temp_env.set_state(check_case.state_with_key.state)
        temp_env.render()

        policy_dict, value = predict(agent, temp_env, network_only=network_only)

        if policy_dict is None:
            logger.warning(f"  Could not get prediction for this state.")
            continue

        if value is not None:
            if check_case.expected_value is None:
                logger.info(f"  Value: Predicted={value:.4f}")
            else:
                logger.info(
                    f"  Value: Expected={check_case.expected_value:.1f}, Predicted={value:.4f}"
                )

        sorted_probs = sorted(
            policy_dict.items(), key=lambda item: item[1], reverse=True
        )

        logger.info(f"  Predicted Probabilities for Legal Actions:")
        if not sorted_probs:
            logger.info("    - (No legal actions)")
        else:
            for action, prob in sorted_probs:
                highlight = ""
                if action == check_case.expected_action:
                    highlight += " <<< EXPECTED"
                if sorted_probs and action == sorted_probs[0][0]:
                    highlight += " (BEST)"
                logger.info(f"    - {action}: {prob:.4f}{highlight}")


def main():
    """
    Runs Connect4 sanity checks on all configured agents.
    """
    config = AppConfig()
    config.env.name = "Connect4"

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("--- Running Sanity Checks for Connect4 ---")

    env = get_environment(config.env)
    agents = get_agents(env, config)

    agents_to_check = []
    for agent_name, agent in agents.items():
        agents_to_check.append((agent_name, agent, False))
        if isinstance(agent, AlphaZeroAgent):
            agents_to_check.append((f"{agent_name} (Network Only)", agent, True))

    for agent_name, agent, network_only in agents_to_check:
        run_sanity_checks_for_agent(env, agent, agent_name, network_only=network_only)


if __name__ == "__main__":
    main()
