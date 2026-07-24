"""Compact robotics primitives for feedback control and motion planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence
import numpy as np

Point = tuple[float, float]


@dataclass
class PDController:
    kp: float
    kd: float
    target: float = 0.0
    previous_error: float | None = None

    def step(self, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = self.target - measurement
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.kd * derivative


def systematic_resample(weights: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("weights must be a non-empty non-negative vector")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    weights = weights / total
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.minimum(indices, len(weights) - 1)


class RRTPlanner:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        collision_free: Callable[[Point, Point], bool],
        step_size: float = 0.5,
        goal_sample_rate: float = 0.1,
        max_iterations: int = 5000,
        seed: int | None = None,
    ) -> None:
        if step_size <= 0 or not 0 <= goal_sample_rate <= 1:
            raise ValueError("invalid planner parameters")
        self.bounds = bounds
        self.collision_free = collision_free
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _sample(self, goal: Point) -> Point:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        xmin, xmax, ymin, ymax = self.bounds
        return (
            float(self.rng.uniform(xmin, xmax)),
            float(self.rng.uniform(ymin, ymax)),
        )

    def _steer(self, source: Point, target: Point) -> Point:
        distance = self._distance(source, target)
        if distance <= self.step_size:
            return target
        scale = self.step_size / distance
        return (
            source[0] + (target[0] - source[0]) * scale,
            source[1] + (target[1] - source[1]) * scale,
        )

    @staticmethod
    def _path(nodes: list[Point], parents: list[int], index: int) -> list[Point]:
        path: list[Point] = []
        while index != -1:
            path.append(nodes[index])
            index = parents[index]
        return list(reversed(path))

    def plan(self, start: Point, goal: Point, goal_tolerance: float = 0.5) -> list[Point] | None:
        nodes = [start]
        parents = [-1]

        for _ in range(self.max_iterations):
            sample = self._sample(goal)
            nearest_index = min(
                range(len(nodes)),
                key=lambda i: self._distance(nodes[i], sample),
            )
            candidate = self._steer(nodes[nearest_index], sample)
            if not self.collision_free(nodes[nearest_index], candidate):
                continue
            nodes.append(candidate)
            parents.append(nearest_index)
            candidate_index = len(nodes) - 1
            if (
                self._distance(candidate, goal) <= goal_tolerance
                and self.collision_free(candidate, goal)
            ):
                nodes.append(goal)
                parents.append(candidate_index)
                return self._path(nodes, parents, len(nodes) - 1)
        return None
