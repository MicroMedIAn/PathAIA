# coding utf-8
"""Useful functions for general use"""


def ifnone(x, default):
    """
    Wrapper for default assignment in case of None.

    Arguments:
        x(any): input to be tested
        default(any): default value for x

    Returns:
        any: x if x is not None else default
    """
    