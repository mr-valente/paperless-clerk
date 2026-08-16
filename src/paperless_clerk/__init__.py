"""Paperless Clerk: local document intelligence for Paperless-ngx."""

import os
from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    try:
        return version("paperless-clerk")
    except PackageNotFoundError:
        return "0.1.0"


def _runtime_version() -> str:
    return os.environ.get("PAPERLESS_CLERK_VERSION") or _installed_version()


__version__ = _runtime_version()
