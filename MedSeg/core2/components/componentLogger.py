import logging
import logging.config
import yaml

# Used from: https://theaisummer.com/logging-debugging/
# Open a Yaml config file
with open("core2/components/log_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def get_logger(name: str = "componentLogger"):
    """
    Logs a message

    Args:
        name(str): name of logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
