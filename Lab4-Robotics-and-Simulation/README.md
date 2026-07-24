# Lab 4 — Robotics and Simulation

## Topics

- Particle-filter localization and resampling
- PD feedback control
- RRT motion planning
- Collision checking and waypoint following

## Public implementation

[`src/ai_labs/robotics.py`](../src/ai_labs/robotics.py) contains independent,
framework-neutral implementations of:

- systematic particle resampling;
- a stateful PD controller;
- a collision-aware RRT planner.

Tests are in [`tests/test_robotics.py`](../tests/test_robotics.py).

## Key takeaway

The lab connects three layers of an autonomous system: localization estimates
the state, planning proposes a feasible route, and feedback control turns the
route into stable motion.
