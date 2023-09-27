from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from nptyping import NDArray, Shape
from scipy.sparse import triu
from sortedcontainers import SortedDict
from tqdm import tqdm

from .object_api import Tree, UGraph
from .types import Edge, Node


class AgglomerativeClustering:
    r"""
    Object used to hierarchically cluster nodes on a graph. Clustering greedily chooses
    to merge linked nodes that have minimum distance/strength ratio. Strength between
    2 nodes is initially 1 for every edge and 0 when there is no edge, then when 2 nodes
    are merged the strength of a newly formed link between the new node and another node
    is the weighted (by node population) average of the strengths between the 2 old
    nodes and the other node. This algorithm uses centroid linkage clustering (UPGMC).

    Args:
        compute_all: whether to initially compute all distances between nodes regardless
            of there linkage.
    """

    def __init__(self, compute_all: bool = False):
        self.compute_all = compute_all

    def init_graph(
        self,
        G: UGraph,
        feats: Union[Dict[Node, NDArray[Shape["*"], Any]], Sequence[str]],
        weights: Optional[Union[Dict[Edge, float], str]] = None,
    ):
        r"""
        Initialize main graph attributes (adjacency matrix, n_nodes and features) using
        a graph object, a list of features and a list of weights.

        Args:
            G: graph to cluster nodes on.
            feats: either a dictionary that maps nodes to their corresponding feature
                vectors or a sequence of property names that will be used as features.
            weights: either a dictionary that maps edges to their corresponding weight
                or a property name that will be used as weight. If `None` is passed,
                weights are computed using euclidian distances between feature vectors.
        """
        self.A = triu(G.A, format="csr").astype(np.float32)
        self.n_nodes = G.n_nodes
        if isinstance(feats, dict):
            feats = [feats[node] for node in G.nodes]
            self.feats = np.stack(feats)
        else:
            self.feats = []
            for node in G.nodes:
                self.feats.append([G.nodeprops[feat][node] for feat in feats])
            self.feats = np.array(self.feats)

        if weights is None:
            ii, jj = self.A.nonzero()
            dists = ((feats[ii] - feats[jj]) ** 2).sum(1)
            self.A[ii, jj] = dists
        elif isinstance(weights, dict):
            for (n1, n2) in weights:
                i, j = sorted((G.nodes.index(n1), G.nodes.index(n2)))
                self.A[i, j] = weights[n1, n2] ** 2
        else:
            for (n1, n2) in G.edges:
                i, j = sorted((G.nodes.index(n1), G.nodes.index(n2)))
                self.A[i, j] = G.edgeprops[str][(n1, n2)] ** 2

    def reset(self):
        """
        Reset the algorithm attributes. Populations are initiated to 1 for every node,
        strengths are initiated to 1 for every edge, dendrogram is emptied.
        """
        self.populations_ = {k: 1 for k in range(self.n_nodes)}
        ii, jj = self.A.nonzero()
        self.centroids_ = {k: self.feats[k] for k in range(self.n_nodes)}
        self.links_ = {
            k: set(jj[ii == k].tolist() + ii[jj == k].tolist())
            for k in range(self.n_nodes)
        }
        self.strengths_ = {(i, j): 1 for i, j in zip(ii, jj)}
        if self.compute_all:
            self.distances_ = {
                (i, j): self.distance(i, j)
                for i in range(self.n_nodes)
                for j in range(i + 1, self.n_nodes)
            }
        else:
            self.distances_ = {(i, j): self.distance(i, j) for i, j in zip(ii, jj)}
        self.edges_ = SortedDict(
            self.criterion, {(i, j): self.distances_[i, j] for i, j in zip(ii, jj)}
        )
        self.dendrogram_ = np.zeros((self.n_nodes - 1, 4))

    def distance(self, i: int, j: int) -> float:
        """
        Get squared distance between nodes `i` and `j`. If available in the adjacency
        matrix or in the `distance` dictionary it is not recomputed.

        Args:
            i: first node.
            j: second node.

        Returns:
            Squared euclidian distance between i and j.
        """
        i, j = sorted((i, j))
        try:
            d = self.A[i, j]
        except IndexError:
            d = self.distances_.get((i, j), 0)
        if not d:
            d = ((self.centroids_[i] - self.centroids_[j]) ** 2).sum()
        return d

    def criterion(self, x: Tuple[int, int]) -> float:
        """
        Criterion function used to find the next nodes to merge. Override it to use
        another criterion.

        Args:
            x: tuple containg the two nodes to merge.

        Returns:
            Squared distance between the 2 nodes divided by link strength.
        """
        i, j = x
        return self.distances_[i, j] / self.strengths_[i, j]

    def create_centroid_link(self, i, j, c, k):
        """
        Create a new link between centroid `c` (that comes from merging nodes `i` and
        `j`) and node `k`.

        Args:
            i: first merged node.
            j: second merged node.
            c: centroid of nodes `i` and `j`.
            k: node linked to either `i` or `j` or both.
        """
        if i == k or j == k:
            return
        ik, ki = sorted((i, k))
        jk, kj = sorted((j, k))
        ck, kc = sorted((c, k))
        ij, ji = sorted((i, j))

        pi = self.populations_[i]
        pj = self.populations_[j]
        ri = pi / (pi + pj)
        rj = pj / (pi + pj)
        try:
            dik = self.distances_[ik, ki]
            djk = self.distances_[jk, kj]
            dij = self.distances_[ij, ji]
            dck = ri * dik + rj * djk - ri * rj * dij
        except KeyError:
            dck = self.distance(c, k)

        self.distances_[ck, kc] = dck

        self.edges_.pop((ik, ki), 0)
        self.edges_.pop((jk, kj), 0)
        sik = self.strengths_.get((ik, ki), 0)
        sjk = self.strengths_.get((jk, kj), 0)
        if sik or sjk:
            self.strengths_[ck, kc] = ri * sik + rj * sjk
            self.edges_[ck, kc] = dck

        self.links_[k].discard(i)
        self.links_[k].discard(j)
        self.links_[k].add(c)
        self.links_[j].discard(k)
        self.links_[c].add(k)

    def add_link(self, i: int, j: int):
        """
        Create a new link between 2 nodes.

        Args:
            i: first node.
            j: second node.
        """
        i, j = sorted((i, j))
        dij = self.distance(i, j)
        self.distances_[i, j] = dij
        self.strengths_[i, j] = 1
        self.edges_[i, j] = dij
        self.links_[i].add(j)
        self.links_[j].add(i)

    def fit(
        self,
        G: UGraph,
        feats: Union[Dict[Node, NDArray[Shape["*"], Any]], Sequence[str]],
        weights: Optional[Union[Dict[Edge, float], str]] = None,
    ):
        r"""
        Fits on the given graph and completes the dendrogram. A dendrogram is an array
        of size :math:`(n-1) \times 4` (whre :math:`n` is the number of nodes)
        representing the successive merges of nodes. Each row gives the two merged
        nodes, their distance and the size of the resulting cluster. Any new node
        resulting from a merge takes the first available index (e.g., the first merge
        corresponds to node :math:`n`).

        Args:
            G: graph to cluster nodes on.
            feats: either a dictionary that maps nodes to their corresponding feature
                vectors or a sequence of property names that will be used as features.
            weights: either a dictionary that maps edges to their corresponding weight
                or a property name that will be used as weight. If `None` is passed,
                weights are computed using euclidian distances between feature vectors.
        """
        self.init_graph(G, feats, weights)
        self.reset()

        c = self.n_nodes
        for n in tqdm(range(self.n_nodes - 1), total=self.n_nodes - 1):
            if not self.edges_:
                cur_dendrogram = self.dendrogram_[:n]
                missing = sorted(
                    [
                        k
                        for k in range(c)
                        if k not in cur_dendrogram[:, 0]
                        and k not in cur_dendrogram[:, 1]
                    ]
                )
                for k, i in enumerate(missing):
                    for j in missing[k + 1 :]:
                        self.add_link(i, j)
            (i, j), _ = self.edges_.popitem(0)

            pi = self.populations_[i]
            pj = self.populations_[j]
            ri = pi / (pi + pj)
            rj = pj / (pi + pj)
            self.dendrogram_[n] = [i, j, self.criterion((i, j)), pi + pj]

            self.centroids_[c] = ri * self.centroids_[i] + rj * self.centroids_[j]
            self.populations_[c] = pi + pj
            self.links_[c] = set()

            while self.links_[i]:
                k = self.links_[i].pop()
                self.create_centroid_link(i, j, c, k)
            while self.links_[j]:
                k = self.links_[j].pop()
                self.create_centroid_link(j, i, c, k)
            self.links_.pop(i)
            self.links_.pop(j)
            c += 1

    def fit_transform(
        self,
        G: UGraph,
        feats: Union[Dict[Node, NDArray[Shape["*"], Any]], Sequence[str]],
        weights: Optional[Union[Dict[Edge, float], str]] = None,
    ) -> Tree:
        """
        Fits on the given graph and returns the hierarchical clustering tree.

        Args:
            G: graph to cluster nodes on.
            feats: either a dictionary that maps nodes to their corresponding feature
                vectors or a sequence of property names that will be used as features.
            weights: either a dictionary that maps edges to their corresponding weight
                or a property name that will be used as weight. If `None` is passed,
                weights are computed using euclidian distances between feature vectors.

        Returns:
            The tree that describes the hierarchical clustering procedure.
        """
        self.fit(G, feats, weights)

        children = []
        parents = []
        nodes = list(range(G.n_nodes))

        if isinstance(weights, str):
            key = weights
        else:
            key = "weight"
        edgeprops = {key: {}}

        if isinstance(feats, dict):
            n_feats = next(iter(feats)).shape
            nodeprops = {
                k: {n: feats[node][k] for n, node in enumerate(G.nodes)}
                for k in range(n_feats)
            }
        else:
            nodeprops = {
                feat: {n: G.nodeprops[feat][node] for n, node in enumerate(G.nodes)}
                for feat in feats
            }
        nodeprops["population"] = {n: 1 for n in range(G.n_nodes)}

        for k, row in enumerate(self.dendrogram_):
            n = k + self.n_nodes
            n1, n2 = row[:2]
            children[n] = [n1, n2]
            parents[n1] = n
            parents[n2] = n
            nodes.append(n)
            if isinstance(feats, dict):
                for k, centroid in enumerate(self.centroids_[n]):
                    nodeprops[k][n] = centroid
            else:
                for k, feat in enumerate(feats):
                    nodeprops[feat][n] = self.centroids_[n, k]
            edgeprops[key][n, n1] = self.distance(n, n1) ** 0.5
            edgeprops[key][n, n2] = self.distance(n, n2) ** 0.5
            nodeprops["population"][n] = self.populations_[n]

        return Tree(nodes, parents, children, nodeprops, edgeprops)
