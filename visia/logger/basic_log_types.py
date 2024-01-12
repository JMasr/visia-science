import logging

from enum import Enum

# Variables of possibles origin of logs
log_origin_main_module: str = "MAIN_MODULE"

# Variables of possibles types of logs
log_type_debug: int = logging.DEBUG
log_type_error: int = logging.ERROR
log_type_info: int = logging.INFO
log_type_warning: int = logging.WARNING


class LogOrigins(Enum):
    """
    Enum class with the possible origins of logs.
    """

    MAIN_MODULE: str = log_origin_main_module


class LogTypes(Enum):
    """
    Enum class with the possible types of logs.
    """

    DEBUG: int = log_type_debug
    ERROR: int = log_type_error
    INFO: int = log_type_info
    WARNING: int = log_type_warning

    def __str__(self):
        """
        Return the name of the enum for a more human-readable log.

        :return: Name of the enum.
        :rtype: str
        """
        return self.name
