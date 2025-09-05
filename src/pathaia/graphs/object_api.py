"""Classes used to represent graphs."""
from typing import List, Sequence, Optional, Union, Tuple
import json
import warnings
from scipy.sparse import spmatrix, dok_matrix
import numpy as np
from ordered_set import OrderedSet
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import translate
from pathlib import Path
from .types import (
    Node,
    NodeProperties,
    BinaryNodeProperty,
    NumericalNodeProperty,
    Parenthood,
    Childhood,
    Edge,
    UEdge,
    EdgeProperties,
    NumericalEdgeProperty,
)
from ..util.types import PathLike
from .errors import (
    InvalidNodeProps,
    UndefinedParenthood,
    UndefinedChildhood,
    UnknownNodeProperty,
)
from .functional_api import (
    complete_tree as _complete_tree,
    get_nodeprops_edgeprops,
    get_root as _get_root,
    get_root_path as _get_root_path,
    get_leaves as _get_leaves,
    tree_to_json as _tree_to_json,
    kruskal_tree as _kruskal_tree,
    cut_on_property as _cut_on_property,
    common_ancestor as _common_ancestor,
    edge_dist as _edge_dist,
    weighted_dist as _weighted_dist,
    get_kneighbors_graph,
)
from ..util.basic import ifnone
import ast

MAX_N_NODES = int(10e7)


class Graph:
    """Object to represent a directed graph."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        edges: Optional[Sequence[Edge]] = None,
        A: Optional[spmatrix] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
    ):
        self.A_ = dok_matrix((MAX_N_NODES, MAX_N_NODES), dtype=bool)
        if nodes is None:
            self.nodes_ = OrderedSet()
            if edges is not None:
                self.edges_ = set(edges)
                for x, y in edges:
                    i = self.nodes_.add(x)
                    j = self.nodes_.add(y)
                    self.A_[i, j] = True
            elif A is not None:
                self.nodes_ = OrderedSet(np.arange(A.shape[0]))
                self.edges_ = set()
                for i, j in zip(*A.nonzero()):
                    self.edges_.add((i, j))
                    self.A_[i, j] = True
            else:
                self.edges_ = set()
        else:
            self.nodes_ = OrderedSet(nodes)
            if edges is not None:
                self.edges_ = set(edges)
                for x, y in edges:
                    i = self.nodes_.index(x)
                    j = self.nodes_.index(y)
                    self.A_[i, j] = True
            elif A is not None:
                self.edges_ = set()
                for i, j in zip(*A.nonzero()):
                    self.edges_.add((self.nodes_[i], self.nodes_[j]))
                    self.A_[i, j] = True
            else:
                self.edges_ = set()

        self.nodeprops = ifnone(nodeprops, {})
        self.edgeprops = ifnone(edgeprops, {})

    @property
    def n_nodes(self):
        return len(self.nodes_)

    @property
    def nodes(self):
        return self.nodes_

    @property
    def edges(self):
        return self.edges_

    @property
    def A(self):
        return self.A_.tocsr()[: self.n_nodes, : self.n_nodes]

    def add_node(self, node: Node):
        self.nodes_.add(node)

    def add_nodes(self, nodes: Sequence[Node]):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, edge: Edge):
        self.add_nodes(edge)
        self.edges_.add(edge)
        n1, n2 = edge
        i = self.nodes_.index(n1)
        j = self.nodes_.index(n2)
        self.A_[i, j] = True

    def add_edges(self, edges: Sequence[Edge]):
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, edge: Edge):
        try:
            self.edges_.remove(edge)
        except KeyError:
            print(f"Edge {edge} was not found in graph")
        n1, n2 = edge
        i = self.nodes_.index(n1)
        j = self.nodes_.index(n2)
        self.A_[i, j] = False

    def reset(self):
        self.nodes_ = OrderedSet()
        self.edges_ = set()
        self.A_ = dok_matrix((MAX_N_NODES, MAX_N_NODES), dtype=bool)
        self.nodeprops = {}
        self.edgeprops = {}


class UGraph(Graph):
    """Class to represent an undirected graph."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        edges: Optional[Sequence[Edge]] = None,
        A: Optional[spmatrix] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
    ):
        super().__init__(nodes, edges, A, nodeprops, edgeprops)
        self.edges_ = {UEdge(edge, key=self.nodes_.index) for edge in self.edges_}

    @property
    def A(self):
        A = self.A_.tocsr()[: self.n_nodes, : self.n_nodes]
        return A + A.T

    def add_edge(self, edge: Edge):
        super().add_edge(UEdge(edge, key=self.nodes_.index))

    def remove_edge(self, edge: Edge):
        super().remove_edge(UEdge(edge, key=self.nodes_.index))
        n1, n2 = edge
        i = self.nodes_.index(n1)
        j = self.nodes_.index(n2)
        self.A_[j, i] = False

    @classmethod
    def from_hovernet_wsi_file(
        cls,
        wsi_file: PathLike,
        n_farthest_samples: Union[int, float] = 0.3,
        n_random_samples: Union[int, float] = 0.1,
        dmax: int = 500,
        n_neighbors: int = 5,
        n_jobs: Optional[int] = None,
    ):
        """
        Create a cell graph from a single hovernet json file generated from their WSI
        script.

        Args:
            wsi_file: json_file generated by hovernet's run_wsi.sh.
            n_farthest_samples: number of points to keep using farthest points sampling.
                If a float is given, represents the proportion of points used instead.
            n_random_samples: number of points to keep using random sampling. If a float
                is given, represents the proportion of points used instead.
            dmax: maximum distance in pixels between two adjacent nodes.
            n_neighbors: number of neighbors to use for KNN algorithm.
            n_jobs: number of parallel jobs to run for neighbors search. None means 1.

        Returns:
            A UGraph representing cell nuclei connections.
        """
        with open(wsi_file, "r") as f:
            nuc_dict = json.load(f)
        centroids = []

        for k in nuc_dict["nuc"]:
            x, y = nuc_dict["nuc"][k]["centroid"]
            centroids.append((x, y))
        centroids = np.array(centroids)

        A = get_kneighbors_graph(
            centroids,
            n_farthest_samples=n_farthest_samples,
            n_random_samples=n_random_samples,
            dmax=dmax,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )
        nodeprops, edgeprops = get_nodeprops_edgeprops(A, centroids)
        return cls(A=A, nodeprops=nodeprops, edgeprops=edgeprops)

    @classmethod
    def from_hovernet_patch_file(
        cls,
        patch_folder: PathLike,
        n_farthest_samples: Union[int, float] = 0.3,
        n_random_samples: Union[int, float] = 0.1,
        dmax: int = 500,
        n_neighbors: int = 5,
        n_jobs: Optional[int] = None,
    ):
        """
        Create a cell graph from a folder containing hovernet json files generated from
        their tile script.

        Args:
            patch_folder: folder containing json_files generated by hovernet's
            run_tile.sh. Files must be named with x_y_level.json formatting.
            n_farthest_samples: number of points to keep using farthest points sampling.
                If a float is given, represents the proportion of points used instead.
            n_random_samples: number of points to keep using random sampling. If a float
                is given, represents the proportion of points used instead.
            dmax: maximum distance in pixels between two adjacent nodes.
            n_neighbors: number of neighbors to use for KNN algorithm.
            n_jobs: number of parallel jobs to run for neighbors search. None means 1.

        Returns:
            A UGraph representing cell nuclei connections.
        """
        patch_folder = Path(patch_folder)
        polygons = []

        for json_file in patch_folder.iterdir():
            with open(json_file, "r") as f:
                nuc_dict = json.load(f)
            x, y = map(int, json_file.stem.split("_")[:2])
            for k in nuc_dict["nuc"]:
                contour = nuc_dict["nuc"][k]["contour"]
                polygon = Polygon(contour)
                polygon = translate(polygon, xoff=x, yoff=y)
                polygons.append(polygon)
        polygons = unary_union(polygons)

        centroids = [(polygon.centroid.x, polygon.centroid.y) for polygon in polygons]
        centroids = np.array(centroids, dtype=np.int32)

        A = get_kneighbors_graph(
            centroids,
            n_farthest_samples=n_farthest_samples,
            n_random_samples=n_random_samples,
            dmax=dmax,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )
        nodeprops, edgeprops = get_nodeprops_edgeprops(A, centroids)
        return cls(A=A, edgeprops=edgeprops, nodeprops=nodeprops)


class Tree(Graph):
    """Object to handle trees."""

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        edges: Optional[Sequence[Edge]] = None,
        parents: Optional[Parenthood] = None,
        children: Optional[Childhood] = None,
        nodeprops: Optional[NodeProperties] = None,
        edgeprops: Optional[EdgeProperties] = None,
        jsonfile: Optional[str] = None,
    ):
        """Init tree object."""
        if jsonfile is not None:
            self.from_json(jsonfile)
            edges = set()
            for parent in self.children_:
                for child in self.children_[parent]:
                    edges.add((parent, child))
        else:
            if edges is not None and (parents is not None or children is not None):
                warnings.warn(
                    "Be careful when specifying both edges and parents/children,"
                    "consistency will not be checked and edges will be prioritized."
                )
            if edges is None:
                edges = set()
                self.parents_, self.children_ = _complete_tree(parents, children)
                for parent in self.children_:
                    for child in self.children_[parent]:
                        edges.add((parent, child))
            else:
                edges = set(edges)
                self.parents_ = {}
                self.children_ = {}
                for parent, child in edges:
                    self.parents_[child] = parent
                    try:
                        self.children_[parent].add(child)
                    except KeyError:
                        self.children_[parent] = {child}
        super().__init__(
            nodes=nodes, edges=edges, nodeprops=nodeprops, edgeprops=edgeprops
        )

    @property
    def parents(self) -> Parenthood:
        return self.parents_

    @property
    def children(self) -> Childhood:
        return self.children_

    def add_edge(self, parent: Node, child: Node):
        self.parents_[child] = parent
        try:
            self.children_[parent].add(child)
        except KeyError:
            self.children_[parent] = {child}
        super().add_edge((parent, child))

    def add_children(self, parent: Node, children: Sequence[Node]):
        for child in children:
            self.parents_[child] = parent
            super().add_edge((parent, child))
        try:
            self.children_[parent] |= set(children)
        except KeyError:
            self.children_[parent] = set(children)

    def add_edges(self, edges: Sequence[Tuple[Node, Union[Node, Sequence[Node]]]]):
        for p, c in edges:
            if isinstance(c, Node):
                self.add_edge(p, c)
            else:
                self.add_children(p, c)

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
