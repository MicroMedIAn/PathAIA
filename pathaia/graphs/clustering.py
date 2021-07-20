from sortedcontainers import SortedDict
from scipy.sparse import triu, csr_matrix
import numpy as np
from tqdm import tqdm
from nptyping import NDArray
from typing import Any, Tuple


class AgglomerativeClustering:
    r"""
    Object used to hierarchically cluster nodes on a graph. Clustering greedily chooses
    to merge linked nodes that have minimum distance/strength ratio. Strength between
    2 nodes is initially 1 for every edge and 0 when there is no edge, then when 2 nodes
    are merged the strength of a newly formed link between the new node and another node
    is the weighted (by node population) average of the strengths between the 2 old
    nodes and the other node. This algorithm uses centroid linkage clustering (UPGMC).

    Args:
        A: adjacency matrix of the graph as a scipy sparse matrix where
        .. math::
                A=
                \begin{cases}
                    \text{dist}_{ij} \text{if i and j are linked}
                    0 \text{else}
                \end{cases}
        feats: feature matrix of size `(n_nodes, n_features)`.
        compute_all: whether to initially compute all distances between nodes regardless
            of there linkage.
    """

    def __init__(
        self,
        A: csr_matrix,
        feats: NDArray[(Any, Any), float],
        compute_all: bool = False,
    ):
        self.A = triu(A.maximum(A.T), format="csr")
        self.n_nodes = A.shape[0]
        self.compute_all = compute_all
        self.feats = feats

    def reset(self):
        """
        Reset the algorithm attributes. Populations are initiated to 1 for every node,
        strengths are initiated to 1 for every edge, dendrogram is emptied.
        """
        self.populations = {k: 1 for k in range(self.n_nodes)}
        ii, jj = self.A.nonzero()
        self.centroids = {k: self.feats[k] for k in range(self.n_nodes)}
        self.links = {
            k: set(jj[ii == k].tolist() + ii[jj == k].tolist())
            for k in range(self.n_nodes)
        }
        self.strengths = {(i, j): 1 for i, j in zip(ii, jj)}
        if self.compute_all:
            self.distances = {
                (i, j): self.distance(i, j)
                for i in range(self.n_nodes)
                for j in range(i + 1, self.n_nodes)
            }
        else:
            self.distances = {(i, j): self.distance(i, j) for i, j in zip(ii, jj)}
        self.edges = SortedDict(
            self.criterion, {(i, j): self.distances[i, j] for i, j in zip(ii, jj)}
        )
        self.dendrogram = np.zeros((self.n_nodes - 1, 4))

    def distance(self, i: int, j: int) -> float:
        """
        Get squared distance between nodes `i` and `j`. If available in the adjacency
        matrix or in the `distance` dictionary it is not recomputed.

        Args:
            i: first node.
            j: second node

        Returns:
            Squared euclidian distance between i and j.
        """
        i, j = sorted((i, j))
        try:
            d = self.A[i, j] ** 2
        except IndexError:
            d = self.distances.get((i, j), 0)
        if not d:
            d = ((self.centroids[i] - self.centroids[j]) ** 2).sum()
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
        return self.distances[i, j] / self.strengths[i, j]

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

        pi = self.populations[i]
        pj = self.populations[j]
        ri = pi / (pi + pj)
        rj = pj / (pi + pj)
        try:
            dik = self.distances[ik, ki]
            djk = self.distances[jk, kj]
            dij = self.distances[ij, ji]
            dck = ri * dik + rj * djk - ri * rj * dij
        except KeyError:
            dck = self.distance(c, k)

        self.distances[ck, kc] = dck

        self.edges.pop((ik, ki), 0)
        self.edges.pop((jk, kj), 0)
        sik = self.strengths.get((ik, ki), 0)
        sjk = self.strengths.get((jk, kj), 0)
        if sik or sjk:
            self.strengths[ck, kc] = ri * sik + rj * sjk
            self.edges[ck, kc] = dck

        self.links[k].discard(i)
        self.links[k].discard(j)
        self.links[k].add(c)
        self.links[j].discard(k)
        self.links[c].add(k)

    def add_link(self, i: int, j: int):
        """
        Create a new link between 2 nodes.

        Args:
            i: first node.
            j: second node.
        """
        i, j = sorted((i, j))
        dij = self.distance(i, j)
        self.distances[i, j] = dij
        self.strengths[i, j] = 1
        self.edges[i, j] = dij
        self.links[i].add(j)
        self.links[j].add(i)

    def fit(self):
        r"""
        Fits on the adjacency matrix and completes the dendrogram. A dendrogram is an
        array of size :math:`(n-1) \times 4` (whre :math:`n` is the number of nodes)
        representing the successive merges of nodes. Each row gives the two merged
        nodes, their distance and the size of the resulting cluster. Any new node
        resulting from a merge takes the first available index (e.g., the first merge
        corresponds to node :math:`n`).
        """
        self.reset()
        c = self.n_nodes
        for n in tqdm(range(self.n_nodes - 1), total=self.n_nodes - 1):
            if not self.edges:
                cur_dendrogram = self.dendrogram[:n]
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
            (i, j), _ = self.edges.popitem(0)

            pi = self.populations[i]
            pj = self.populations[j]
            ri = pi / (pi + pj)
            rj = pj / (pi + pj)
            self.dendrogram[n] = [i, j, self.criterion((i, j)), pi + pj]

            self.centroids[c] = ri * self.centroids[i] + rj * self.centroids[j]
            self.populations[c] = pi + pj
            self.links[c] = set()

            while self.links[i]:
                k = self.links[i].pop()
                self.create_centroid_link(i, j, c, k)
            while self.links[j]:
                k = self.links[j].pop()
                self.create_centroid_link(j, i, c, k)
            self.links.pop(i)
            self.links.pop(j)
            c += 1
