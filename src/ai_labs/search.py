"""Generic graph-search algorithms with a small, framework-independent API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from itertools import count
from typing import Callable, Hashable, Iterable, Protocol, TypeVar

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action")


class SearchProblem(Protocol[State, Action]):
    @property
    def start(self) -> State: ...

    def is_goal(self, state: State) -> bool: ...

    def successors(self, state: State) -> Iterable[tuple[State, Action, float]]: ...


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...]
    cost: float
    expanded: int


def _reconstruct(
    goal: State,
    parent: dict[State, tuple[State, Action]],
    cost: float,
    expanded: int,
) -> SearchResult:
    actions: list[Action] = []
    state = goal
    while state in parent:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return SearchResult(tuple(actions), cost, expanded)


def depth_first_search(problem: SearchProblem[State, Action]) -> SearchResult | None:
    frontier: list[tuple[State, float]] = [(problem.start, 0.0)]
    parent: dict[State, tuple[State, Action]] = {}
    discovered = {problem.start}
    expanded = 0

    while frontier:
        state, path_cost = frontier.pop()
        if problem.is_goal(state):
            return _reconstruct(state, parent, path_cost, expanded)
        expanded += 1
        for next_state, action, step_cost in problem.successors(state):
            if next_state not in discovered:
                discovered.add(next_state)
                parent[next_state] = (state, action)
                frontier.append((next_state, path_cost + step_cost))
    return None


def breadth_first_search(problem: SearchProblem[State, Action]) -> SearchResult | None:
    frontier = deque([(problem.start, 0.0)])
    parent: dict[State, tuple[State, Action]] = {}
    discovered = {problem.start}
    expanded = 0

    while frontier:
        state, path_cost = frontier.popleft()
        if problem.is_goal(state):
            return _reconstruct(state, parent, path_cost, expanded)
        expanded += 1
        for next_state, action, step_cost in problem.successors(state):
            if next_state not in discovered:
                discovered.add(next_state)
                parent[next_state] = (state, action)
                frontier.append((next_state, path_cost + step_cost))
    return None


def _best_first_search(
    problem: SearchProblem[State, Action],
    heuristic: Callable[[State], float],
) -> SearchResult | None:
    serial = count()
    frontier: list[tuple[float, int, float, State]] = [
        (heuristic(problem.start), next(serial), 0.0, problem.start)
    ]
    parent: dict[State, tuple[State, Action]] = {}
    best_cost = {problem.start: 0.0}
    expanded = 0

    while frontier:
        _, _, path_cost, state = heapq.heappop(frontier)
        if path_cost != best_cost.get(state):
            continue
        if problem.is_goal(state):
            return _reconstruct(state, parent, path_cost, expanded)
        expanded += 1
        for next_state, action, step_cost in problem.successors(state):
            if step_cost < 0:
                raise ValueError("UCS and A* require non-negative edge costs")
            next_cost = path_cost + step_cost
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                parent[next_state] = (state, action)
                priority = next_cost + heuristic(next_state)
                heapq.heappush(
                    frontier,
                    (priority, next(serial), next_cost, next_state),
                )
    return None


def uniform_cost_search(problem: SearchProblem[State, Action]) -> SearchResult | None:
    return _best_first_search(problem, heuristic=lambda _: 0.0)


def a_star(
    problem: SearchProblem[State, Action],
    heuristic: Callable[[State], float],
) -> SearchResult | None:
    return _best_first_search(problem, heuristic)
