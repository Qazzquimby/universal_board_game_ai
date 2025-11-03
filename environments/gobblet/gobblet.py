"""
Resembles 4x4 tic tac toe with different piece sizes and the ability to move
pieces already on the board, with bigger pieces covering smaller pieces.

Two colors, black and white.
Each has a *reserve* of 3 piles of 4 pieces with sizes 1-4, stacked.
Pieces always stack with larger on top of smaller. They don't need to be consecutive in size.
Only the top piece of a stack can ever be moved, whether on the board or reserve. Lower pieces stay where they were.

A player can either
- move a piece from their *reserve* onto an empty space or onto one of your own pieces,
- move a piece on the board orthogonally one space

If a player ever has a 4-in-a-row at the end of their turn, they win.
Its possible for both players to have a 4-in-a-row simultaneously if a piece moves off another pieces,
in which case the currently acting player still wins.
"""
