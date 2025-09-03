# TODO
# See if train_alphzero.py can be easily modified to train muzero as well.

# get_hidden_state takes state (state tokens, game token, action tokens)
# Runs through transformer encoder
# Makes vector hidden_state = (state tokens mean) concat (state tokens max) concat (game token)

# get_hidden_state_actions takes (hidden state vector)
# makes list of fully encoded action tokens (representing action tokens that have already been through the transformer encoder)
# Ideally these are made in parallel rather than autoregressively, but whichever works better.

# get_next_hidden_state takes (hidden state vector, action token)
# makes next hidden vector state