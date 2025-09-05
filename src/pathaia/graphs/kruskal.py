from .types import Parenthood, NumericalNodeProperty, Node, Edge


class UFDS:
    """
    Union-find data structure for felzenszwalb algorithm.
    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:
    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.
    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py
    """

    def __init__(self):
        """
        Create a new empty union-find structure.
        ****************************************
        """
        # UFDS permit to compute the hierarchy
        # it is responsible to keep track of:
        # the parent is crucial, but it only guarantee the tree structure
        # should not be used outside of the object
        self._parent: Parenthood = {}
        # size is subjective, it is only used to fuse objects, no real quantitative interest
        # should not be used outside of the object
        self._size: NumericalNodeProperty = {}

    def get_root(self, node: Node) -> Node:
        """
        Find and return the name of the set containing the node.
        ********************************************************
        """
        # check for previously unknown object
        # if unknown, create an entry in the parent dico
        # then, size is set to 1
        # intra is set to 0
        if node not in self._parent:
            self._parent[node] = node
            self._size[node] = 1
            return node
        # find path of objects leading to the root
        path = [node]
        root = self._parent[node]
        while root != path[-1]:
            path.append(root)
            root = self._parent[root]
        # compress the path and return
        for ancestor in path:
            self._parent[ancestor] = root
        return root

    def __iter__(self):
        """
        Iterate through all items ever found or unioned by this structure.
        ******************************************************************
        """
        return iter(self.parents)

    def union(self, edge: Edge):
        """
        Find the sets containing the objects and merge them all.
        ********************************************************
        """
        n1, n2 = edge
        roots = [self.get_root(n1), self.get_root(n2)]
        # Find the heaviest root according to its weight.
        heaviest = max(roots, key=lambda r: self._size[r])
        for r in roots:
            if r != heaviest:
                self._size[heaviest] += self._size[r]
                self._parent[r] = heaviest
