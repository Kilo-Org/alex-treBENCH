"""
Unit tests for ConfigValidator.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from core.config_validator import ConfigValidator
from core.exceptions import ConfigurationError


class TestConfigValidator:
    """Test cases for ConfigValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = ConfigValidator()
        self.test_config_path = Path("test_config.yaml")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_config_path.exists():
            self.test_config_path.unlink()

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config_data = {
            'app': {
                'name': 'Test App',
                'version': '1.0.0'
            },
            'database': {
                'url': 'sqlite:///test.db'
            },
            'benchmark': {
                'default_sample_size': 1000,
                'confidence_level': 0.95
            },
            'logging': {
                'level': 'INFO',
                'file': 'test.log'
            },
            'kaggle': {
                'dataset': 'test/dataset',
                'cache_dir': 'test/cache'
            }
        }

        result = self.validator.validate_config(config_data)
        assert result == config_data

    def test_validate_config_missing_required_field(self):
        """Test validation with missing required field."""
        config_data = {
            'app': {
                'name': 'Test App'
                # Missing version
            },
            'database': {
                'url': 'sqlite:///test.db'
            }
        }

        with pytest.raises(ConfigurationError, match="Missing required field: app.version"):
            self.validator.validate_config(config_data)

    def test_validate_config_missing_required_section(self):
        """Test validation with missing required section."""
        config_data = {
            'app': {
                'name': 'Test App',
                'version': '1.0.0'
            }
            # Missing database section
        }

        with pytest.raises(ConfigurationError, match="Missing required section: database"):
            self.validator.validate_config(config_data)

    def test_validate_version_valid(self):
        """Test valid version validation."""
        self.validator._validate_version("1.0.0")
        self.validator._validate_version("2.1.3")

    def test_validate_version_invalid(self):
        """Test invalid version validation."""
        with pytest.raises(ValueError, match="Version must be in format x.y.z"):
            self.validator._validate_version("1.0")

        with pytest.raises(ValueError, match="Version must be in format x.y.z"):
            self.validator._validate_version("1.0.0-beta")

        with pytest.raises(ValueError, match="Version must be in format x.y.z"):
            self.validator._validate_version("invalid")

    def test_validate_positive_int_valid(self):
        """Test valid positive integer validation."""
        self.validator._validate_positive_int(1)
        self.validator._validate_positive_int(100)

    def test_validate_positive_int_invalid(self):
        """Test invalid positive integer validation."""
        with pytest.raises(ValueError, match="Must be a positive integer"):
            self.validator._validate_positive_int(0)

        with pytest.raises(ValueError, match="Must be a positive integer"):
            self.validator._validate_positive_int(-1)

        with pytest.raises(ValueError, match="Must be a positive integer"):
            self.validator._validate_positive_int("not_a_number")

    def test_validate_non_negative_int_valid(self):
        """Test valid non-negative integer validation."""
        self.validator._validate_non_negative_int(0)
        self.validator._validate_non_negative_int(1)
        self.validator._validate_non_negative_int(100)

    def test_validate_non_negative_int_invalid(self):
        """Test invalid non-negative integer validation."""
        with pytest.raises(ValueError, match="Must be a non-negative integer"):
            self.validator._validate_non_negative_int(-1)

        with pytest.raises(ValueError, match="Must be a non-negative integer"):
            self.validator._validate_non_negative_int("not_a_number")

    def test_validate_confidence_level_valid(self):
        """Test valid confidence level validation."""
        self.validator._validate_confidence_level(0.8)
        self.validator._validate_confidence_level(0.95)
        self.validator._validate_confidence_level(0.99)

    def test_validate_confidence_level_invalid(self):
        """Test invalid confidence level validation."""
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.validator._validate_confidence_level(0)

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.validator._validate_confidence_level(1.5)

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.validator._validate_confidence_level("not_a_number")

    def test_validate_margin_of_error_valid(self):
        """Test valid margin of error validation."""
        self.validator._validate_margin_of_error(0.05)
        self.validator._validate_margin_of_error(0.1)

    def test_validate_margin_of_error_invalid(self):
        """Test invalid margin of error validation."""
        with pytest.raises(ValueError, match="Margin of error must be between 0 and 0.5"):
            self.validator._validate_margin_of_error(0)

        with pytest.raises(ValueError, match="Margin of error must be between 0 and 0.5"):
            self.validator._validate_margin_of_error(0.6)

    def test_validate_similarity_threshold_valid(self):
        """Test valid similarity threshold validation."""
        self.validator._validate_similarity_threshold(0.0)
        self.validator._validate_similarity_threshold(0.5)
        self.validator._validate_similarity_threshold(1.0)

    def test_validate_similarity_threshold_invalid(self):
        """Test invalid similarity threshold validation."""
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            self.validator._validate_similarity_threshold(-0.1)

        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            self.validator._validate_similarity_threshold(1.1)

    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            'app': {
                'name': 'Base App',
                'version': '1.0.0'
            },
            'database': {
                'url': 'sqlite:///base.db'
            }
        }

        override_config = {
            'app': {
                'name': 'Override App'
            },
            'logging': {
                'level': 'DEBUG'
            }
        }

        merged = self.validator.merge_configs(base_config, override_config)

        assert merged['app']['name'] == 'Override App'
        assert merged['app']['version'] == '1.0.0'
        assert merged['database']['url'] == 'sqlite:///base.db'
        assert merged['logging']['level'] == 'DEBUG'

    @patch('core.config_validator.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_validate_and_load_success(self, mock_file, mock_exists):
        """Test successful configuration loading and validation."""
        mock_exists.return_value = True
        mock_config_data = {
            'app': {'name': 'Test App', 'version': '1.0.0'},
            'database': {'url': 'sqlite:///test.db'},
            'benchmark': {'default_sample_size': 1000, 'confidence_level': 0.95},
            'logging': {'level': 'INFO', 'file': 'test.log'},
            'kaggle': {'dataset': 'test/dataset', 'cache_dir': 'test/cache'}
        }
        mock_file.return_value.read.return_value = yaml.dump(mock_config_data)

        config = self.validator.validate_and_load(self.test_config_path)

        assert config.app.name == 'Test App'
        assert config.database.url == 'sqlite:///test.db'

    @patch('core.config_validator.Path.exists')
    def test_validate_and_load_file_not_found(self, mock_exists):
        """Test loading configuration when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            self.validator.validate_and_load(self.test_config_path)

    @patch('core.config_validator.os.environ')
    def test_validate_environment_config(self, mock_environ):
        """Test environment variable configuration validation."""
        mock_environ.get.return_value = None
        mock_environ.__iter__.return_value = []

        result = self.validator.validate_environment_config()
        assert result == {}

    def test_validate_environment_config_with_values(self, monkeypatch):
        """Test environment variable configuration with actual values."""
        monkeypatch.setenv('JB_DATABASE_URL', 'sqlite:///env.db')
        monkeypatch.setenv('JB_LOG_LEVEL', 'DEBUG')

        result = self.validator.validate_environment_config('JB_')

        assert result['database.url'] == 'sqlite:///env.db'
        assert result['logging.level'] == 'DEBUG'

    def test_validate_environment_config_invalid_value(self, monkeypatch):
        """Test environment variable configuration with invalid value."""
        monkeypatch.setenv('JB_APP_VERSION', 'invalid.version')

        with pytest.raises(ConfigurationError, match="Environment configuration errors"):
            self.validator.validate_environment_config('JB_')