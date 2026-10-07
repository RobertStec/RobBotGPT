import logging
from logging.config import dictConfig

from core.config import settings


def configure_logging() -> None:
    """
    Configure application-wide logging.
    """

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s | "
                    "%(levelname)s | "
                    "%(name)s | "
                    "%(message)s"
                )
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level,
            }
        },

        "root": {
            "handlers": ["console"],
            "level": settings.log_level,
        },
    })


def get_logger(
    name: str,
) -> logging.Logger:
    return logging.getLogger(name)