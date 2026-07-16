import unittest
import threading
from swarm_backend.core.swarm_state import SwarmState

class TestSwarmState(unittest.TestCase):
    def setUp(self):
        self.state = SwarmState()

    def test_basic_updates(self):
        self.state.update_position("drone_1", (10.0, 20.0, -5.0))
        self.state.update_heading("drone_1", 90.0)

        self.assertEqual(self.state.get_position("drone_1"), (10.0, 20.0, -5.0))
        self.assertEqual(self.state.get_heading("drone_1"), 90.0)
        self.assertEqual(self.state.get_position("drone_2"), (0.0, 0.0, 0.0))

    def test_all_positions_except(self):
        self.state.update_position("leader", (0.0, 0.0, 0.0))
        self.state.update_position("drone_1", (5.0, 5.0, 0.0))
        self.state.update_position("drone_2", (-5.0, 5.0, 0.0))

        others = self.state.all_positions_except("drone_1")
        self.assertEqual(len(others), 2)
        self.assertIn((0.0, 0.0, 0.0), others)
        self.assertIn((-5.0, 5.0, 0.0), others)
        self.assertNotIn((5.0, 5.0, 0.0), others)

    def test_concurrent_safety(self):
        # Fire up multiple threads updating state concurrently
        num_threads = 50
        threads = []

        def worker(idx):
            agent_id = f"drone_{idx}"
            self.state.update_position(agent_id, (float(idx), 0.0, 0.0))
            self.state.update_heading(agent_id, float(idx * 5))

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check that all updates registered safely without crash/corruption
        for i in range(num_threads):
            agent_id = f"drone_{i}"
            self.assertEqual(self.state.get_position(agent_id), (float(i), 0.0, 0.0))
            self.assertEqual(self.state.get_heading(agent_id), float(i * 5))

    def test_reset(self):
        self.state.update_position("leader", (1.0, 2.0, 3.0))
        self.state.reset()
        self.assertEqual(self.state.get_position("leader"), (0.0, 0.0, 0.0))
