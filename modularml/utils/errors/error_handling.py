from enum import Enum


class ErrorMode(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"
    COERCE = "coerce"
