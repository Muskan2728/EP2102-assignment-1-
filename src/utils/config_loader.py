"""Configuration loader utility."""
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config, *keys, default=None):
    """Get nested configuration value.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to access
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
