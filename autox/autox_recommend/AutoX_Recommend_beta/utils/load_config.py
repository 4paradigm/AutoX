import json
from utils.logger import logger


def get_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error('File', config_path, 'Not Found')
    return config
