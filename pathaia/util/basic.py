# coding utf-8
"""Useful functions for general use"""
from typing import Any


def ifnone(x: Any, default: Any) -> Any:
    return x if x is not None else default


def dumb():
    """
    Dumb function.

    A dumb function to test github actions on changes.
    """
    pass
