# Lab 1 — Search and Game Playing

## Topics

- Depth-first search and breadth-first search
- Uniform-cost search and A* search
- Heuristic design
- Minimax, alpha–beta pruning and Monte Carlo tree search

## Public implementation

The reusable search implementation is in
[`src/ai_labs/search.py`](../src/ai_labs/search.py), with tests in
[`tests/test_search.py`](../tests/test_search.py).

The original lab used the UC Berkeley Pacman framework. Its license explicitly
prohibits publishing solutions, so the submitted Pacman files are not included.
The public implementation uses a new generic `SearchProblem` interface and
independent code.

## Key takeaway

For BFS/DFS, a state can usually be marked as discovered when inserted into the
frontier. UCS and A* need a best-known-cost map instead: a newly discovered
cheaper path must be allowed to replace the old one.
