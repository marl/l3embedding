import logging
import logging.handlers


def init_file_logger(logger, log_path=None):
    """
    Initializes logging to a file.

    Saves log to "audiosetdl.log" in the current directory, and rotates them
    after they reach 1MiB.

    Args:
        logger:  Logger object
                 (Type: logging.Logger)
    """
    # Set up file handler
    if not log_path:
        log_path = './l3embedding.log'
    handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=2**20)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def init_console_logger(logger, verbose=False):
    """
    Initializes logging to stdout

    Args:
        logger:  Logger object
                 (Type: logging.Logger)

    Kwargs:
        verbose:  If true, prints verbose information to stdout. False by default.
                  (Type: bool)
    """
    # Log to stderr also
    stream_handler = logging.StreamHandler()
    if verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
