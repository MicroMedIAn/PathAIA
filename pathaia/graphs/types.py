from typing import Union, Dict, List, Tuple

Node = Union[int, str, tuple]
Edge = Tuple[Node, Node]

NodeEndomorphism = Dict[Node, Node]
Parenthood = Dict[Node, Node]
Childhood = Dict[Node, List[Node]]

BinaryNodeProperty = Dict[Node, bool]
NumericalNodeProperty = Dict[Node, Union[float, int]]
SymbolicNodeProperty = Dict[Node, Union[str]]

BinaryEdgeProperty = Dict[Edge, bool]
NumericalEdgeProperty = Dict[Edge, Union[float, int]]
SymbolicEdgeProperty = Dict[Edge, Union[str]]

NodeProperty = Union[SymbolicNodeProperty, NumericalNodeProperty]
EdgeProperty = Union[SymbolicEdgeProperty, NumericalEdgeProperty]

NodeProperties = Dict[str, NodeProperty]
EdgeProperties = Dict[str, EdgeProperty]
