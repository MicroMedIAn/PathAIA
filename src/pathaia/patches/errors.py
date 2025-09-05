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


class PathaiaWarning(Warning):
    """
    Base of custom warnings.

    ************************
    """
    pass


class HasNoDataFolder(PathaiaWarning):
    """
    Raise when trying to read mrxs with no data folder.

    ***************************************************
    """

    pass


class InvalidArgument(Error):
    """
    Raise when trying to read mrxs with no data folder.

    ***************************************************
    """

    pass
