"""
Configuration management for Islamabad Smog Detection System.

This module handles loading, validating, and accessing configuration
parameters from the YAML config file and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages system configuration with environment variable support."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        try:
            # Load environment variables from .env file if exists
            load_dotenv()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()

            # Replace environment variables in config
            config_content = self._substitute_env_vars(config_content)

            # Parse YAML
            self._config = yaml.safe_load(config_content)

            logger.info(f"Configuration loaded from {self.config_path}")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _substitute_env_vars(self, content: str) -> str:
        """
        Replace ${VAR_NAME} placeholders with environment variable values.

        Args:
            content: String content with environment variable placeholders

        Returns:
            Content with environment variables substituted
        """
        import re

        def replace_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not found, using empty string")
                return ""
            return env_value

        # Replace ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, content)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'apis.copernicus.client_id')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'apis', 'region')

        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})

    def validate_required(self) -> bool:
        """
        Validate that required configuration values are present.

        Returns:
            True if all required values are present
        """
        required_keys = [
            'region.center_lat',
            'region.center_lon',
            'region.bounding_box',
            'data_sources.sentinel5p.collections',
            'data_sources.modis.products'
        ]

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False

        return True

    def get_region_bounds(self) -> tuple:
        """
        Get region bounding box coordinates.

        Returns:
            Tuple of (west, south, east, north) coordinates
        """
        bbox = self.get('region.bounding_box')
        if bbox:
            return (bbox['west'], bbox['south'], bbox['east'], bbox['north'])
        else:
            raise ValueError("Region bounding box not configured")

    def get_utm_zone(self) -> int:
        """Get UTM zone for the region."""
        return self.get('region.utm_zone', 43)

    def get_data_collection_info(self, source: str, collection_type: str) -> Dict[str, Any]:
        """
        Get collection information for a specific data source.

        Args:
            source: Data source name ('sentinel5p', 'modis', 'google_earth_engine')
            collection_type: Type of collection to get info for

        Returns:
            Dictionary with collection information
        """
        section = f'data_sources.{source}'
        if collection_type == 'collections':
            return self.get(f'{section}.collections', {})
        elif collection_type == 'products':
            return self.get(f'{section}.products', {})
        else:
            return self.get(section, {})

    def get_processing_params(self, process_type: str) -> Dict[str, Any]:
        """
        Get processing parameters for a specific process type.

        Args:
            process_type: Type of processing ('atmospheric_correction', 'noise_reduction', etc.)

        Returns:
            Processing parameters dictionary
        """
        return self.get(f'processing.{process_type}', {})

    def is_development_mode(self) -> bool:
        """Check if system is in development mode."""
        return self.get('development.debug', False)

    def is_test_mode(self) -> bool:
        """Check if system is in test mode."""
        return self.get('development.test_mode', False)

    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Get API configuration for a specific service.

        Args:
            api_name: Name of API service ('copernicus', 'nasa', 'google_earth_engine')

        Returns:
            API configuration dictionary
        """
        config = self.get(f'apis.{api_name}', {})

        # Validate required fields for certain APIs
        if api_name == 'copernicus':
            required = ['client_id', 'client_secret', 'base_url']
            for field in required:
                if not config.get(field):
                    logger.warning(f"Missing required field {field} in {api_name} API config")

        elif api_name == 'nasa':
            required = ['earthdata_username', 'earthdata_password']
            for field in required:
                if not config.get(field):
                    logger.warning(f"Missing required field {field} in {api_name} API config")

        return config

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.get('storage', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance and parallelization configuration."""
        return self.get('performance', {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None


# Global configuration instance
_config_manager = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration instance.

    Args:
        config_path: Optional path to config file

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def reset_config() -> None:
    """Reset global configuration instance (useful for testing)."""
    global _config_manager
    _config_manager = None