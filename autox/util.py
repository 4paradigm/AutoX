import warnings
warnings.filterwarnings('ignore')

# log
import logging
LOGGER = logging.getLogger('run-time-adaptive_automl')
LOG_LEVEL = 'INFO'
# LOG_LEVEL = 'DEBUG'
LOGGER.setLevel(getattr(logging, LOG_LEVEL))
simple_formatter = logging.Formatter('%(levelname)7s -> %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(simple_formatter)
LOGGER.addHandler(console_handler)
LOGGER.propagate = False
nesting_level = 0

def log(entry, level='info'):
    if level not in ['debug', 'info', 'warning', 'error']:
        LOGGER.error('Wrong level input')

    global nesting_level
    space = '-' * (4 * nesting_level)

    getattr(LOGGER, level)(f"{space} {entry}")