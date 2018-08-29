import time
import logging
import logging.handlers


class LogTimer(object):
    """
    A context manager that times the execution of a block of code and logs it.
    """

    def __init__(self, logger, desc, log_level=logging.DEBUG):
        """
        Creates an instance of a log timer context manager.

        The log events will be logged to the given logger with the given level
        in the format:

        '<desc>' took <duration> seconds


        Args:
            logger:     Logger to log to
                        (Type: logging.Logger)

            desc:       Description of the code to time
                        (Type: str)

            log_level:  Logging level to log to. By default, the level is
                        set to logging.DEBUG.
                        (Type: int, logging.{NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL})
        """
        ## Logger used to log timing information
        ## (Type: logging.Logger)
        self.logger = logger
        ## Logging level to log to
        ## (Type: int, logging.{NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL})
        self.log_level = log_level
        ## Description of the code block being timed
        ## (Type: str)
        self.desc = desc
        self._start_time = None

        if log_level == logging.NOTSET:
            raise ValueError('Cannot use NOTSET logging level.')

    def __enter__(self):
        """
        Get the execution start time before the code block starts executing.
        """
        self._start_time = time.time()

    def __exit__(self, type_, value, tb):
        """
        Compute the runtime duration of the code block, and log it to the given logger.

        If an error occurred during the code block, nothing is logged.


        Args:
            type_:  Type of Exception thrown, if one was thrown before the
                    context was exited.
                    (Type: type)

            value:  The instance of the Exception thrown if one was thrown
                    before the context was exited.
                    (Type: Exception)

            tb:     The traceback of the error that occurred.
                    (Type: str)
        """
        # Compute the duration that the block of code took to execute
        end_time = time.time()
        duration = end_time - self._start_time
        self._start_time = None

        # We don't need to time if something went wrong
        if type_ or value or tb:
            return

        # Make the message out of the given description and the duration
        msg = "{0} took {1} seconds".format(self.desc, duration)

        # Log at the appropriate level
        if self.log_level == logging.DEBUG:
            self.logger.debug(msg)
        elif self.log_level == logging.INFO:
            self.logger.info(msg)
        elif self.log_level == logging.WARNING:
            self.logger.warning(msg)
        elif self.log_level == logging.ERROR:
            self.logger.error(msg)
        elif self.log_level == logging.CRITICAL:
            self.logger.critical(msg)


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
        stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
