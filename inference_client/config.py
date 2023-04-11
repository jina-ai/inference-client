import os

from pydantic import BaseSettings

# logging
DEFAULT_LOGGING_LEVEL = 'INFO'
DEFAULT_LOGGING_FORMAT = '[%(name)-32s]  %(message)s'
DEFAULT_LOGGING_DATE_FORMAT = '[%X]'
DEFAULT_LOGGER_NAME = 'inference-client'


# api endpoint
DEFAULT_API_ENDPOINT = 'https://api.clip.jina.ai/api/v1'


class Settings(BaseSettings):
    """Settings for the inference client."""

    logging_level: str = DEFAULT_LOGGING_LEVEL
    logging_format: str = DEFAULT_LOGGING_FORMAT
    logging_date_format: str = DEFAULT_LOGGING_DATE_FORMAT
    logger_name: str = DEFAULT_LOGGER_NAME

    api_endpoint: str = DEFAULT_API_ENDPOINT

    class Config:
        env_file = os.environ.get('CLIENT_ENV_FILE', '.env')


settings = Settings()
