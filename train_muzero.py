# Context

# A state is made of several dataframes, where each row is converted to a token.
# Actions are another set of tokens
# Muzero will need to generate a set of action tokens in order to MCTS.
# You can *not* assume a single finite superset of actions or a fixed count of actions.
# The game logs contain the states for each turn of a game, in order.
