import unittest

from ai_labs.search import a_star, breadth_first_search, uniform_cost_search


class TinyGraph:
    start = "S"
    edges = {
        "S": [("A", "S-A", 2), ("B", "S-B", 1)],
        "A": [("G", "A-G", 2)],
        "B": [("G", "B-G", 10)],
        "G": [],
    }

    def is_goal(self, state):
        return state == "G"

    def successors(self, state):
        return self.edges[state]


class SearchTests(unittest.TestCase):
    def test_bfs_finds_shallow_path(self):
        result = breadth_first_search(TinyGraph())
        self.assertEqual(result.actions, ("S-A", "A-G"))

    def test_ucs_finds_low_cost_path(self):
        result = uniform_cost_search(TinyGraph())
        self.assertEqual(result.cost, 4)
        self.assertEqual(result.actions, ("S-A", "A-G"))

    def test_astar_finds_low_cost_path(self):
        heuristic = {"S": 3, "A": 2, "B": 1, "G": 0}
        result = a_star(TinyGraph(), heuristic.__getitem__)
        self.assertEqual(result.cost, 4)


if __name__ == "__main__":
    unittest.main()
