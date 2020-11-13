# coding: utf8
"""
A module to handle patchification errors.

Enable catching of level and filter accesses.
"""


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class UnknownLevelError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class UnknownFilterError(Error):
    """
    Raise when trying to use an unknown filter.

    *********************************************
    """

    pass
