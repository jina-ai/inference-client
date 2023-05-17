import os

from .__version__ import __version__
from .client import Client

__all__ = ["__version__", "Client"]

if 'NO_VERSION_CHECK' not in os.environ:
    from .__version__ import is_latest_version

    is_latest_version(github_repo='inference-client')
