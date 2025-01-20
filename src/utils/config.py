import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)
def load_config(filepath="config.yaml"):
    """Load configuration from a YAML file."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {filepath}.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}.")
        raise

def get_config_value(config, section, key):
    """Retrieves a value from the config, with error handling."""
    try:
        return config[section][key]
    except KeyError as e:
        logger.error(f"Key {key} not found in section {section} of the config.")
        raise
