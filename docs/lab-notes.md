# Lab learning notes

## Lab 1 — Search and game playing

Implemented and compared uninformed and informed graph search. The central
design decision was when a state should enter the visited set: marking on
insertion avoids duplicate frontier entries for BFS/DFS, while cost-aware
algorithms need a best-known-cost map and must allow a cheaper path to update a
state. I also practiced minimax, alpha–beta pruning and Monte Carlo tree search
in an adversarial environment.

Public scope: this repository contains a new generic search API only. It does
not publish the course's Pacman solution code.

## Lab 2 — Machine learning from scratch

Implemented logistic regression, tree-based models and the forward/backward
passes of a small neural network using NumPy. The most transferable lesson was
that mathematical correctness is not sufficient: stable sigmoid and log-loss
implementations, shape discipline, initialization and regularization determine
whether a model trains reliably.

The public module focuses on a small, tested logistic-regression implementation
to keep the example inspectable.

## Lab 3 — NLP and an LLM-controlled agent

Built transparent NLP baselines including multinomial Naive Bayes, word-vector
processing and TF–IDF retrieval, then used a language model as a decision
component in a constrained environment. The exercise highlighted the boundary
between probabilistic baselines and LLM systems: API credentials must remain
outside source control, output formats need validation, and model actions must
be checked against the environment's legal-action set.

The public repository contains no API key and makes no external model call.

## Lab 4 — Robotics and simulation

Practiced particle-filter localization, PD feedback control and RRT motion
planning. These algorithms connect estimation, control and planning:

1. localization estimates where the agent is;
2. the planner proposes a feasible route;
3. the controller turns route targets into stable motion commands.

The public module provides reusable resampling, PD-control and RRT primitives
without distributing the original simulator or assignment tests.
