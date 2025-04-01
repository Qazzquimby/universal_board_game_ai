The goal is to create a universal board game engine which makes it easy (as it can be) to define any
board game, up to and including extremely complex board games like MtG.
The engine then automatically designs, parameterizes, and trains a powerful AI model to play the
game.

The current focus is on validating the AI generation part of the process.
The current unvalidated plan is to

- represent the gamestate as a heterogeneous graph
- use stochastic muzero with graph transformers
- pass the legal actions in as part of the game state rather than having a fixed policy network

# Validation roadmap

- [x] create evaluation environment and reusable base classes.
- [ ] implement alphazero to fair performance
- [ ] implement muzero
- [ ] create graph versions of environments and zero models and iterate until performance is okay
- [ ] instead of using fixed policy output, pass in legal actions as part of the gamestate and have
  variable sized policy output. Iterate until performance is okay.
- [ ] gradually work with more complex games

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
