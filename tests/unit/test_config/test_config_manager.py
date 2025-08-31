"""
Unit tests for ConfigManager.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml
import json

from core.config_manager import ConfigManager, get_config_manager
from core.config import AppConfig
from core.exceptions import ConfigurationError


class TestConfigManager:
    """Test cases for ConfigManager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config_manager = ConfigManager()
        self.test_config_path = Path("test_config.yaml")
        self.test_env_config_path = Path("config/test_env.yaml")

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up any created files
        for path in [self.test_config_path, self.test_env_config_path]:
            if path.exists():
                path.unlink()

    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_get_config_manager_function(self):
        """Test get_config_manager function returns singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2

    @patch('core.config_manager.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_config_success(self, mock_file, mock_exists):
        """Test successful configuration loading."""
        mock_exists.return_value = True
        mock_config_data = {
            'app': {'name': 'Test App', 'version': '1.0.0'},
            'database': {'url': 'sqlite:///test.db'}
        }
        mock_file.return_value.read.return_value = yaml.dump(mock_config_data)

        config = self.config_manager.load_config(self.test_config_path)

        assert isinstance(config, AppConfig)
        assert config.app.name == 'Test App'

    @patch('core.config_manager.Path.exists')
    def test_load_config_file_not_found(self, mock_exists):
        """Test loading configuration when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            self.config_manager.load_config(self.test_config_path)

    def test_get_config_without_loading(self):
        """Test getting config when none is loaded."""
        # Reset the global config
        self.config_manager._config = None

        # Should load default config
        config = self.config_manager.get_config()
        assert isinstance(config, AppConfig)

    def test_set_environment(self):
        """Test switching environment configurations."""
        # This would require setting up test environment files
        # For now, test the error case
        with pytest.raises(ConfigurationError, match="Environment 'nonexistent' not found"):
            self.config_manager.set_environment('nonexistent')

    @patch('core.config_manager.yaml.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_export_config_yaml(self, mock_file, mock_yaml_dump):
        """Test exporting configuration to YAML."""
        # Setup a mock config
        self.config_manager._config = AppConfig()

        output_path = Path("export_test.yaml")
        self.config_manager.export_config(output_path, "yaml")

        mock_yaml_dump.assert_called_once()
        mock_file.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    def test_export_config_json(self, mock_file):
        """Test exporting configuration to JSON."""
        # Setup a mock config
        self.config_manager._config = AppConfig()

        output_path = Path("export_test.json")
        self.config_manager.export_config(output_path, "json")

        # Verify file was opened for writing
        mock_file.assert_called_once_with(output_path, 'w')

    def test_export_config_invalid_format(self):
        """Test exporting with invalid format."""
        self.config_manager._config = AppConfig()

        with pytest.raises(ConfigurationError, match="Unsupported export format"):
            self.config_manager.export_config(Path("test.txt"), "invalid")

    def test_update_config_value(self):
        """Test updating configuration values."""
        # Setup initial config
        self.config_manager._config = AppConfig()

        # Update a value
        self.config_manager.update_config_value('app.name', 'Updated Name')

        assert self.config_manager._config.app.name == 'Updated Name'

    def test_get_config_value(self):
        """Test getting configuration values."""
        # Setup initial config
        self.config_manager._config = AppConfig()
        self.config_manager._config.app.name = 'Test Name'

        # Get the value
        value = self.config_manager.get_config_value('app.name')
        assert value == 'Test Name'

    def test_get_config_value_nested(self):
        """Test getting nested configuration values."""
        # Setup initial config
        self.config_manager._config = AppConfig()
        self.config_manager._config.database.url = 'sqlite:///test.db'

        # Get nested value
        value = self.config_manager.get_config_value('database.url')
        assert value == 'sqlite:///test.db'

    def test_get_config_value_not_found(self):
        """Test getting non-existent configuration value."""
        self.config_manager._config = AppConfig()

        value = self.config_manager.get_config_value('nonexistent.key', 'default')
        assert value == 'default'

    def test_validate_current_config_valid(self):
        """Test validating valid configuration."""
        self.config_manager._config = AppConfig()

        # Mock the validator to return True
        with patch.object(self.config_manager, '_validator') as mock_validator:
            mock_validator.validate_config.return_value = {}

            result = self.config_manager.validate_current_config()
            assert result is True

    def test_validate_current_config_invalid(self):
        """Test validating invalid configuration."""
        self.config_manager._config = AppConfig()

        # Mock the validator to raise an exception
        with patch.object(self.config_manager, '_validator') as mock_validator:
            mock_validator.validate_config.side_effect = ConfigurationError("Invalid config")

            result = self.config_manager.validate_current_config()
            assert result is False

    def test_get_config_info(self):
        """Test getting configuration information."""
        self.config_manager._config = AppConfig()
        self.config_manager._config_path = Path("test_config.yaml")

        info = self.config_manager.get_config_info()

        assert info['config_path'] == 'test_config.yaml'
        assert info['environment'] == 'production'
        assert 'is_valid' in info

    def test_list_environments(self):
        """Test listing available environments."""
        # Create a test environment file
        env_dir = Path("config")
        env_dir.mkdir(exist_ok=True)
        test_env_file = env_dir / "test_env.yaml"
        test_env_file.write_text("app:\n  name: Test Env\n")

        try:
            # Re-scan config files
            self.config_manager._scan_config_files()

            envs = self.config_manager.list_environments()
            assert 'test_env' in envs
        finally:
            # Clean up
            if test_env_file.exists():
                test_env_file.unlink()

    def test_list_models(self):
        """Test listing available model configurations."""
        # Create test model config
        models_dir = Path("config/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        test_model_file = models_dir / "test_model.yaml"
        test_model_file.write_text("models:\n  defaults:\n    temperature: 0.7\n")

        try:
            # Re-scan config files
            self.config_manager._scan_config_files()

            models = self.config_manager.list_models()
            assert 'test_model' in models
        finally:
            # Clean up
            if test_model_file.exists():
                test_model_file.unlink()