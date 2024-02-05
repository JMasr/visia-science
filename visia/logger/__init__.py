"""
Description
============

This is a module for logging events and metrics of Visia-Science.
It contains a custom logger class with the following features:

- Set a log policies.
- Set the default log level.
- Log messages to the console and a file.

Create a logger::

    logger = BasicLogger(log_folder='logs').get_logger()

    logger = CustomLogger(log_folder='logs', console_log_level=logging.INFO).get_logger()

    logger = CustomLogger(log_folder='logs', max_log_file_size=5 * 1024 * 1024, backup_count=3).get_logger()


Log some messages::

    logger.info('This is an info message')

    logger.debug('This is a debug message')

    logger.warning('This is a warning message')

    logger.error('This is an error message')

"""

import logging
import os
from logging.handlers import RotatingFileHandler


class BasicLogger:
    """
    Basic logger class that logs messages to the console and a file.

    :param log_folder: Path to the log folder.
    :type   log_folder: str
    :param log_name: Name of the logger.
    :type   log_name: str
    :param log_file_path: Path to the log file.
    :type   log_file_path: str
    :param error_log_file_path: Path to the error log file.
    :type   error_log_file_path: str
    :param max_log_size: Maximum log file size.
    :type   max_log_size: int
    :param backup_count: Number of log files to keep.
    :type   backup_count: int
    :return: Basic logger object.
    :rtype: BasicLogger

    """

    def __init__(
            self,
            log_folder: str,
            log_name: str = "Visia-Scince_Logger",
            log_file_path="app.log",
            error_log_file_path="error.log",
            max_log_size: int = (5 * 1024 * 1024),
            backup_count: int = 3,
            log_level=logging.DEBUG,
    ):
        # Create the root logger
        self.logger = logging.getLogger(log_name)

        # Set the log policy
        self.log_level = log_level
        self.logger.setLevel(logging.DEBUG)
        self.max_log_size = max_log_size
        self.backup_count = backup_count

        # Set the path to the logs
        self.log_folder = log_folder
        self.log_file_path = os.path.join(log_folder, log_file_path)
        self.error_log_file_path = os.path.join(log_folder, error_log_file_path)

        # Create a formatter to add the time, name, level and message of the log
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a file handler to store logs in a file
        os.makedirs(log_folder, exist_ok=True)
        file_handler = RotatingFileHandler(
            self.log_file_path, maxBytes=self.max_log_size, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create a rotating file handler for error logs
        self.error_file_handler = RotatingFileHandler(
            self.error_log_file_path,
            maxBytes=self.max_log_size,
            backupCount=backup_count,
        )
        self.error_file_handler.setLevel(logging.ERROR)
        self.error_file_handler.setFormatter(formatter)
        self.logger.addHandler(self.error_file_handler)

        # Create a stream handler to print logs in the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_log_level(self, log_level: int):
        """
        Set the log level.

        :param log_level: Log level to use.
        :type log_level: int
        :return: None
        """
        self.logger.setLevel(log_level)

    def get_logger(self):
        """
        Returns the logger object with the specified configuration.

        :return: Logger object.
        :rtype: logging.Logger
        """
        return self.logger
