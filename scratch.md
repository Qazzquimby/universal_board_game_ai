- Force overfitting on one game to see if loss becomes near 0.
- Learn only value
- Learn only policy
- Plot loss curves with wandb


# Blog

Weaknesses of the dominion model, and goal of full generality

MuZero, more flexible, less work for the programmer, more work for the machine learning
Maximally general. state as graph
Sampling state, sampling transitions, progressive widening
Action tokens

Staged training and other efficiency stuff

# Not problems

Yes, the Vae sampling and action tokens are not needed for connect 4.
We want the model to not be crippled by its increased generality. It should learn not to use flexibility it doesn't need. We'd hardly expect it to thrive at complex games if it fails at simple games.

Loss discount scaling is intentional.
Consistency loss is proven to improve training speed. It may be disabled for simplicity but isn't a big concern.