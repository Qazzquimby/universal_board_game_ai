# Using a structure similar to eval, run two instances of mcts many times.
# The intention is to collect game states for model training, similar to alphazero training but off of regular mcts.

# states can be represented by a string of actions taken to get to that state
# eg, R playing in column 0, then Y playing in column 4 = "04"
# each state must record the next action that was taken from that state, and the the final game result
# 0 if player 0 wins, 1 if player 1 wins.

# Rather than starting games at an empty board, have players take 5 random moves each (which should not be recorded).

# save the states as a json file to /data/connect_4_states/pure_mcts
