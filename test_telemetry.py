import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to sys.path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.executors.swarm_executor import connect_drone
from web_dashboard.app import pipeline_state

class TestTelemetry(unittest.IsolatedAsyncioTestCase):
    
    @patch('src.executors.swarm_executor.System')
    async def test_connect_drone_happy_path(self, mock_system_class):
        """Test that connect_drone connects to the correct MAVSDK port for a given UDP port (happy path)."""
        mock_drone = MagicMock()
        mock_system_class.return_value = mock_drone
        mock_drone.connect = AsyncMock()
        
        # Mock connection state stream
        async def mock_connection_state():
            yield MagicMock(is_connected=True)
            
        mock_drone.core.connection_state.return_value = mock_connection_state()
        
        # Mock health stream
        async def mock_health():
            yield MagicMock(
                is_global_position_ok=True,
                is_home_position_ok=True,
                is_armable=True
            )
            
        mock_drone.telemetry.health.return_value = mock_health()
        
        # Call for drone 2 (port 14542 -> index 2 -> MAVSDK port 50053)
        drone = await connect_drone(14542, timeout_s=2.0)
        
        # Check constructor call
        mock_system_class.assert_called_once_with(port=50053)
        # Check connect call
        mock_drone.connect.assert_called_once_with(system_address="udp://:14542")
        self.assertEqual(drone, mock_drone)

    @patch('src.executors.swarm_executor.System')
    async def test_connect_drone_timeout_edge_case(self, mock_system_class):
        """Test that connect_drone correctly times out and raises RuntimeError if the drone is not ready (edge case)."""
        mock_drone = MagicMock()
        mock_system_class.return_value = mock_drone
        mock_drone.connect = AsyncMock()
        
        # Mock connection state stream that never yields connected or hangs
        async def mock_connection_state():
            while True:
                yield MagicMock(is_connected=False)
                await asyncio.sleep(0.1)
            
        mock_drone.core.connection_state.return_value = mock_connection_state()
        
        # Check that TimeoutError triggers RuntimeError
        with self.assertRaises(RuntimeError) as context:
            await connect_drone(14541, timeout_s=0.2)
            
        self.assertIn("did not become ready within", str(context.exception))

if __name__ == '__main__':
    unittest.main()
