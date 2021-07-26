"""Classes used to represent graphs."""
from typing import List, Sequence, Optional, Union
import json
from scipy.sparse import csr_matrix
import numpy as np
from ordered_set import OrderedSet
from .types import (
    Node,
    NodeProperties,
    BinaryNodeProperty,
    NumericalNodeProperty,
    Parenthood,
    Childhood,
    Edge,
    EdgeProperties,
    NumericalEdgeProperty,
)
from .errors import (
    InvalidNodeProps,
    UndefinedParenthood,
    UndefinedChildhood,
    UnknownNodeProperty,
)
from .functional_api import (
    get_root as _get_root,
    get_root_path as _get_root_path,
    get_leaves as _get_leaves,
    tree_to_json as _tree_to_json,
    kruskal_tree as _kruskal_tree,
    cut_on_property as _cut_on_property,
    common_ancestor as _common_ancestor,
    edge_dist as _edge_dist,
    weighted_dist as _weighted_dist,
)
from ..util.basic import ifnone
import ast


class Graph:
    """Object to represent a directed graph."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        edges: Optional[Sequence[Edge]] = None,
        A: Optional[csr_matrix] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
    ):
        if nodes is None:
            self.nodes_ = OrderedSet()
            if edges is not None:
                self.edges_ = list(edges)
                row_ind = []
                col_ind = []
                for x, y in edges:
                    i = self.nodes_.add(x)
                    j = self.nodes_.add(y)
                    row_ind.append(i)
                    col_ind.append(j)
                self.n_nodes = len(self.nodes_)
                self.A_ = csr_matrix(
                    (np.ones(len(row_ind), dtype=bool), (row_ind, col_ind)),
                    shape=(self.n_nodes, self.n_nodes),
                )
            elif A is not None:
                self.nodes_ = OrderedSet(np.arange(A.shape[0]))
                self.n_nodes = len(self.nodes_)
                self.edges_ = [(i, j) for i, j in zip(*A.nonzero())]
                self.A_ = A.astype(bool)
            else:
                self.edges_ = []
                self.A_ = csr_matrix((0, 0), dtype=bool)
        else:
            self.nodes_ = OrderedSet(nodes)
            self.n_nodes = len(self.nodes_)
            if edges is not None:
                self.edges_ = list(edges)
                row_ind = []
                col_ind = []
                for x, y in edges:
                    i = self.nodes_.index(x)
                    j = self.nodes_.index(y)
                    row_ind.append(i)
                    col_ind.append(j)
                self.A_ = csr_matrix(
                    (np.ones(len(row_ind), dtype=bool), (row_ind, col_ind)),
                    shape=(self.n_nodes, self.n_nodes),
                )
            elif A is not None:
                self.edges_ = [(i, j) for i, j in zip(*A.nonzero())]
                self.A_ = A.astype(bool)
            else:
                self.edges_ = []
                self.A_ = csr_matrix((self.n_nodes, self.n_nodes), dtype=bool)

        self.nodeprops = ifnone(nodeprops, {})
        self.edgeprops = ifnone(edgeprops, {})

    @property
    def nodes(self):
        return self.nodes_

    @property
    def edges(self):
        return self.edges_

    @property
    def A(self):
        return self.A_

    def add_node(self, node: Node, update_A: bool = True):
        i = self.nodes_.add(node)
        if i == self.n_nodes:
            self.n_nodes += 1
            if update_A:
                ii, jj = self.A_.nonzero()
                self.A_ = csr_matrix(
                    (self.A_[ii, jj].A1, (ii, jj)),
                    shape=(self.n_nodes, self.n_nodes),
                    dtype=bool,
                )

    def add_nodes(self, nodes: Sequence[Node], update_A: bool = True):
        for node in nodes:
            self.add_node(node, update_A=False)
        if update_A:
            ii, jj = self.A_.nonzero()
            self.A_ = csr_matrix(
                (self.A_[ii, jj].A1, (ii, jj)),
                shape=(self.n_nodes, self.n_nodes),
                dtype=bool,
            )

    def add_edge(self, edge: Edge, update_A: bool = True):
        self.add_nodes(edge, update_A=update_A)
        self.edges_.append(edge)
        if update_A:
            n1, n2 = edge
            i = self.nodes_.index(n1)
            j = self.nodes_.index(n2)
            self.A_[i, j] = True

    def add_edges(self, edges: Sequence[Edge], update_A: bool = True):
        row_ind = []
        col_ind = []
        for edge in edges:
            self.add_edge(edge, update_A=False)
            if update_A:
                n1, n2 = edge
                i = self.nodes_.index(n1)
                j = self.nodes_.index(n2)
                row_ind.append(i)
                col_ind.append(j)
        if update_A:
            ii, jj = self.A_.nonzero()
            data = np.ones((len(ii) + len(row_ind)))
            row_ind = np.concatenate((ii, row_ind))
            col_ind = np.concatenate((jj, col_ind))
            self.A_ = csr_matrix(
                (data, (row_ind, col_ind)),
                shape=(self.n_nodes, self.n_nodes),
                dtype=bool,
            )

    def reset(self):
        self.nodes_ = OrderedSet()
        self.edges_ = []
        self.A_ = csr_matrix((0, 0), dtype=bool)
        self.nodeprops = {}
        self.edgeprops = {}
        self.n_nodes = 0


class UGraph(Graph):
    """Class to represent an undirected graph."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        edges: Optional[Sequence[Edge]] = None,
        A: Optional[csr_matrix] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
    ):
        super().__init__(self, nodes, edges, A, nodeprops, edgeprops)
        self.A_ = A.maximum(A.T)
        self.edges_ = [sorted(edge) for edge in self.edges_]

    def add_edge(self, edge: Edge, update_A: bool = True):
        super().add_edge(sorted(edge), update_A=update_A)
        if update_A:
            self.A_ = self.A_.maximum(self.A_.T)

    def add_edges(self, edges: Sequence[Edge], update_A: bool = True):
        super().add_edges(edges, update_A=update_A)
        if update_A:
            self.A_ = self.A_.maximum(self.A_.T)


class Tree(Graph):
    """Object to handle trees."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        parents: Optional[Parenthood] = None,
        children: Optional[Childhood] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
        jsonfile: Optional[str] = None,
    ):
        """Init tree object."""
        if jsonfile is not None:
            self.from_json(jsonfile)
        else:
            edges = []
            self.parents_ = ifnone(parents, {})
            self.children_ = ifnone(children, {})
            for node in self.children_:
                self.children_[node] = set(self.children_[node])
                edges.extend([(node, child) for child in self.children_[node]])
            super().__init__(
                self, nodes=nodes, edges=edges, nodeprops=nodeprops, edgeprops=edgeprops
            )

    @property
    def parents(self) -> Parenthood:
        return self.parents_

    @property
    def children(self) -> Childhood:
        return self.children_

    def add_edge(self, parent: Node, child: Node, update_A: bool = True):
        self.parents_[child] = parent
        try:
            self.children_[parent].add(child)
        except KeyError:
            self.children_[parent] = {child}
        super().add_edge((parent, child), update_A=update_A)

    def add_children(
        self, parent: Node, children: Sequence[Node], update_A: bool = True
    ):
        edges = []
        for child in children:
            self.parents_[child] = parent
            edges.append((parent, child))
        try:
            self.children_[parent] |= children
        except KeyError:
            self.children_[parent] = set(children)
        super().add_edges(edges, update_A=update_A)

    def add_edges(
        self, edges: Sequence[Node, Union[Node, Sequence[Node]]], update_A: bool = True
    ):
        for p, c in edges:
            if isinstance(c, Node):
                self.add_edge(p, c, update_A=update_A)
            else:
                self.add_children(p, c, update_A=update_A)

    def reset(self):
        super().reset()
        self.parents_ = {}
        self.children_ = {}

    def get_root(self, node: Node = None) -> Node:
        """Give root of the tree."""
        if self.parents_ is not None:
            return _get_root(self.parents_, node)
        raise UndefinedParenthood(
            "Parenthood of the tree was not defined, "
            "please build the tree before use."
        )

    def get_root_path(self, node: Node) -> List[Node]:
        """Get path to root of the tree."""
        if self.parents_ is not None:
            return _get_root_path(self.parents_, node)
        raise UndefinedParenthood(
            "Parenthood of the tree was not defined, "
            "please build the tree before use."
        )

    def get_leaves(
        self, node: Node, prop: Optional[BinaryNodeProperty] = None
    ) -> List[Node]:
        """Get leaves of a node."""
        if self.children_ is not None:
            return _get_leaves(self.children_, node, prop)
        raise UndefinedChildhood(
            "Childhood of the tree was not defined, "
            "please build the tree before use."
        )

    def to_json(self, jsonfile):
        """Store the tree to json file."""
        _tree_to_json(
            self.nodes_,
            self.parents_,
            self.children_,
            jsonfile,
            self.nodeprops,
            self.edgeprops,
        )

    def from_json(self, jsonfile):
        """Create the tree from a json file."""
        # Keep in mind that json keys have to be str.
        # In treez framework, they can be python object as well
        # We use ast to parse the str to a python object before

        # This behaviour might limit even more the types of
        # parenthood/childhood/props keys when using treez...
        with open(jsonfile, "r") as jf:
            json_dict = json.load(jf)
        self.reset()
        for parent, children in json_dict["children"].items():
            try:
                parentkey = ast.literal_eval(parent)
                self.add_children(parentkey, children)
            except (ValueError, SyntaxError):
                self.add_children(parent, children)
        self.edgeprops = dict()
        for name, edgeprop in json_dict["edgeprops"].item():
            self.edgeprops[name] = dict()
            for edgein, edgeout in edgeprop.items():
                try:
                    edgekey = ast.literal_eval(edgein)
                    self.edgeprops[name][edgekey] = edgeout
                except (ValueError, SyntaxError):
                    self.edgeprops[name][edgein] = edgeout
        for name, nodeprop in json_dict["nodeprops"].items():
            for nodein, nodeout in nodeprop.items():
                try:
                    nodekey = ast.literal_eval(nodein)
                    self.nodeprops[name][nodekey] = nodeout
                except (ValueError, SyntaxError):
                    self.nodeprops[name][nodein] = nodeout

    def build_kruskal(
        self,
        edges: Sequence[Edge],
        weights: NumericalEdgeProperty,
        size: NumericalNodeProperty,
    ):
        """Build tree with kruskal algorithm from graph edges."""
        _, k_children, k_props = _kruskal_tree(edges, weights, size)
        for parent in k_children:
            self.add_children(parent, k_children)
        self.nodeprops = k_props

    def cut_on_property(self, cut_name: str, prop: str, threshold: Union[int, float]):
        """
        Produce a list of authorized nodes given a property threshold.
        Set a new property to these nodes.
        """
        if prop in self.nodeprops:
            node_of_interest = _cut_on_property(
                self.parents_, self.children_, self.nodeprops[prop], threshold
            )
            cut = dict()
            for node in self.nodes:
                if node in node_of_interest:
                    cut[node] = True
                else:
                    cut[node] = False
            self.nodeprops[cut_name] = cut

        else:
            raise UnknownNodeProperty(
                "Property {}"
                " is not in the tree properties: {}".format(
                    prop, list(self.nodeprops.keys())
                )
            )

    def common_ancestor(self, node1: Node, node2: Node) -> Node:
        """Return the common ancestor of node1 and node2."""
        return _common_ancestor(self.parents_, node1, node2)

    def edge_dist(self, node1: Node, node2: Node) -> int:
        """Return the number of edges to go from node1 to node2 (by common ancestor)."""
        return _edge_dist(self.parents_, node1, node2)

    def weighted_dist(
        self, weights: Union[NumericalNodeProperty, str], node1: Node, node2: Node
    ) -> float:
        """Return the number of edges to go from node1 to node2 (by common ancestor)."""
        if isinstance(weights, str):
            if weights in self.nodeprops:
                return _weighted_dist(
                    self.parents_, self.nodeprops[weights], node1, node2
                )
            raise InvalidNodeProps(
                "Property {} is not in tree properties: {}".format(
                    weights, self.nodeprops
                )
            )
        if isinstance(weights, dict):
            return _weighted_dist(self.parents_, weights, node1, node2)
        raise InvalidNodeProps(
            "Provided property is not a valid property. "
            "Expected {} or {}, got {}".format(dict, str, type(weights))
        )
