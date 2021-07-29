class Error(Exception):
    """
    Base of custom errors.
    **********************
    """

    pass


class InvalidNodeProps(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class InvalidEdgeProps(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class UndefinedParenthood(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class UndefinedChildhood(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class InvalidTree(Error):
    """
    Raise when parents and children do no match.
    *********************************************
    """


class UnknownNodeProperty(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class InvalidNodeId(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class UnrelatedNode(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass


class UnreachableAncestor(Error):
    """
    Raise when trying to access unknown level.
    *********************************************
    """

    pass
