"""Run a compact demonstration of the four public modules."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_labs.ml import BinaryLogisticRegression
from ai_labs.nlp import TfidfRetriever
from ai_labs.robotics import PDController, RRTPlanner
from ai_labs.search import a_star


class Grid:
    start = (0, 0)
    goal = (3, 2)

    def is_goal(self, state):
        return state == self.goal

    def successors(self, state):
        x, y = state
        for dx, dy, action in ((1, 0, "R"), (-1, 0, "L"), (0, 1, "U"), (0, -1, "D")):
            candidate = (x + dx, y + dy)
            if 0 <= candidate[0] <= 3 and 0 <= candidate[1] <= 2:
                yield candidate, action, 1.0


def main() -> None:
    grid = Grid()
    heuristic = lambda s: abs(s[0] - grid.goal[0]) + abs(s[1] - grid.goal[1])
    route = a_star(grid, heuristic)
    print("A* route:", "".join(route.actions), "cost:", route.cost)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    classifier = BinaryLogisticRegression(learning_rate=0.3, epochs=3000).fit(x, y)
    print("Logistic regression predictions:", classifier.predict(x).tolist())

    retriever = TfidfRetriever().fit(
        {
            "search": "A star combines path cost and a heuristic",
            "robotics": "RRT explores continuous configuration space",
        }
    )
    print("TF-IDF top result:", retriever.rank("heuristic path search", top_k=1)[0])

    controller = PDController(kp=1.0, kd=0.1, target=1.0)
    print("PD outputs:", [round(controller.step(value, 1.0), 3) for value in (0.0, 0.5, 0.9)])

    planner = RRTPlanner(
        bounds=(0, 5, 0, 5),
        collision_free=lambda _a, _b: True,
        seed=7,
        goal_sample_rate=0.3,
    )
    path = planner.plan((0.0, 0.0), (4.0, 4.0))
    print("RRT waypoints:", len(path) if path else 0)


if __name__ == "__main__":
    main()
