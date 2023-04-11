import sys

if sys.version_info < (3, 10):
    # compatibility for python <3.10
    import importlib_metadata as metadata
else:
    from importlib import metadata


def get_version() -> str:
    """Return the module version number specified in pyproject.toml.

    :return: The version number.
    """
    return metadata.version(__package__)


__version__ = get_version()
