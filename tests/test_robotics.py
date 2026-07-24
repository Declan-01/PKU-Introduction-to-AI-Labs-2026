import unittest
import numpy as np

from ai_labs.robotics import PDController, RRTPlanner, systematic_resample


class RoboticsTests(unittest.TestCase):
    def test_pd_controller_reduces_output_as_error_shrinks(self):
        controller = PDController(kp=1.0, kd=0.1, target=1.0)
        first = controller.step(measurement=0.0, dt=1.0)
        second = controller.step(measurement=0.8, dt=1.0)
        self.assertGreater(first, second)

    def test_systematic_resample_prefers_heavy_particle(self):
        rng = np.random.default_rng(0)
        indices = systematic_resample([0.05, 0.05, 0.9], rng)
        self.assertGreaterEqual(np.sum(indices == 2), 2)

    def test_rrt_in_open_space(self):
        planner = RRTPlanner(
            bounds=(0, 5, 0, 5),
            collision_free=lambda _a, _b: True,
            step_size=0.5,
            goal_sample_rate=0.3,
            max_iterations=500,
            seed=7,
        )
        path = planner.plan((0.0, 0.0), (4.0, 4.0))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0.0, 0.0))
        self.assertEqual(path[-1], (4.0, 4.0))


if __name__ == "__main__":
    unittest.main()
