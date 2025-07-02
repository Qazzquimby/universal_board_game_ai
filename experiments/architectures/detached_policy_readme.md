The idea is to add another kind of model based on the transformer architecture which can support
games with dynamic policy spaces.

For example, a game might have a massive growing collection of cards, and we want to query for
"which of these arbitrary card embeddings is best to gain" or discard, or play, in different heads.
There may be many heads for different kinds of choices (choosing from modal options, picking a
location, etc etc)

To handle this, we split the model before the policy head, and make a separate policy modal.
The state model takes the state and outputs encoded tokens.
The policy model is then called once per legal move to give that move a score.
The policy model is given an input of some tokens from the state model (eg the game token, currently
acting token, a mean and max of selected tokens), and potentially other inputs.
We then read a head of the policy corresponding to the kind of choice being made (eg how much it'd
want to discard the selected cards, how much it wants to choose action 3 of the selected card)
We choose the action with the highest score.

We want to be able to train both models together, end to end. We have a list of values and policies
as with any other model.
We're currently working only with connect4, which will have a standard looking policy with only one
index per column, but for sake of testing the architecture we want to make sure the code would be
able to handle choosing from distinct action heads to get the action score.