import logging

from .config import settings

# Set-up the logger
logging.basicConfig(
    level=settings.logging_level,
    format=settings.logging_format,
    datefmt=settings.logging_date_format,
)
logger = logging.getLogger(settings.logger_name)
