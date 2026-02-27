"""Shared fixtures for tests."""

import logging
import pytest


@pytest.fixture(autouse=True)
def _setup_logger():
    """Ensure the lshdp logger exists for all tests."""
    logger = logging.getLogger("lshdp")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    yield
