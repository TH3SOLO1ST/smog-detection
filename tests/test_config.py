"""
Configuration module tests for Islamabad Smog Detection System.
"""

import unittest
import tempfile
import os
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.test_config = {
            'apis': {
                'copernicus': {
                    'client_id': 'test_client_id',
                    'client_secret': 'test_client_secret'
                },
                'nasa': {
                    'earthdata_username': 'test_user',
                    'earthdata_password': 'test_pass'
                }
            },
            'region': {
                'name': 'Islamabad',
                'center_lat': 33.6844,
                'center_lon': 73.0479,
                'buffer_km': 50,
                'bounding_box': {
                    'north': 34.1844,
                    'south': 33.1844,
                    'east': 73.9479,
                    'west': 72.1479
                }
            }
        }

        # Write test config to temporary file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_config_initialization(self):
        """Test configuration manager initialization."""
        config_manager = ConfigManager(str(self.config_file))
        self.assertIsNotNone(config_manager)

    def test_get_config_value(self):
        """Test getting configuration values."""
        config_manager = ConfigManager(str(self.config_file))

        # Test nested config access
        client_id = config_manager.get('apis.copernicus.client_id')
        self.assertEqual(client_id, 'test_client_id')

        # Test default value
        nonexistent = config_manager.get('nonexistent.key', 'default')
        self.assertEqual(nonexistent, 'default')

    def test_get_section(self):
        """Test getting configuration sections."""
        config_manager = ConfigManager(str(self.config_file))

        apis_section = config_manager.get_section('apis')
        self.assertIsInstance(apis_section, dict)
        self.assertIn('copernicus', apis_section)

    def test_region_configuration(self):
        """Test region configuration."""
        config_manager = ConfigManager(str(self.config_file))

        bbox = config_manager.get_region_bounds()
        self.assertEqual(len(bbox), 4)
        self.assertEqual(bbox, (72.1479, 33.1844, 73.9479, 34.1844))

    def test_environment_variable_substitution(self):
        """Test environment variable substitution."""
        # Set environment variable
        os.environ['TEST_VAR'] = 'substituted_value'

        test_config_with_env = {
            'test_section': {
                'test_key': '${TEST_VAR}'
            }
        }

        env_config_file = Path(self.temp_dir) / 'env_config.yaml'
        with open(env_config_file, 'w') as f:
            yaml.dump(test_config_with_env, f)

        config_manager = ConfigManager(str(env_config_file))
        substituted_value = config_manager.get('test_section.test_key')

        # Clean up
        del os.environ['TEST_VAR']

        self.assertEqual(substituted_value, 'substituted_value')


if __name__ == '__main__':
    unittest.main()