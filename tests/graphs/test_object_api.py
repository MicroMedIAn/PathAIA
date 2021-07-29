from pathaia.graphs import Graph, UGraph, Tree
from pathaia.graphs.errors import InvalidTree
from scipy.sparse import csr_matrix
from pytest import raises, warns
import itertools
from ordered_set import OrderedSet


def test_graph_init():
    nodes_init = (None, (1, 2, 3), "123")
    edges_init = (None, ((1, 3), (1, 2), (2, 1)))
    A_init = (None, csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool))

    expected = (
        (OrderedSet(), set(), csr_matrix((0, 0), dtype=bool)),
        (
            OrderedSet((0, 1, 2)),
            {(0, 1), (1, 2)},
            csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 3, 2)),
            {(1, 3), (1, 2), (2, 1)},
            csr_matrix(((1, 1, 1), ((0, 0, 2), (1, 2, 0))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 3, 2)),
            {(1, 3), (1, 2), (2, 1)},
            csr_matrix(((1, 1, 1), ((0, 0, 2), (1, 2, 0))), shape=(3, 3), dtype=bool),
        ),
        (OrderedSet((1, 2, 3)), set(), csr_matrix((3, 3), dtype=bool)),
        (
            OrderedSet((1, 2, 3)),
            {(1, 2), (2, 3)},
            csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 2, 3)),
            {(1, 3), (1, 2), (2, 1)},
            csr_matrix(((1, 1, 1), ((0, 0, 1), (2, 1, 0))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 2, 3)),
            {(1, 3), (1, 2), (2, 1)},
            csr_matrix(((1, 1, 1), ((0, 0, 1), (2, 1, 0))), shape=(3, 3), dtype=bool),
        ),
        (OrderedSet(("1", "2", "3")), set(), csr_matrix((3, 3), dtype=bool)),
        (
            OrderedSet(("1", "2", "3")),
            {("1", "2"), ("2", "3")},
            csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool),
        ),
        KeyError,
        KeyError,
    )

    for exp, (nodes, edges, A) in zip(
        expected, itertools.product(nodes_init, edges_init, A_init)
    ):
        if isinstance(exp, tuple):
            exp_nodes, exp_edges, exp_A = exp
            G = Graph(nodes, edges, A)
            assert G.nodes == exp_nodes
            assert G.edges == exp_edges
            assert (G.A[G.A > 0].A == exp_A[exp_A > 0].A).all()
        else:
            assert raises(exp, Graph, nodes, edges, A)


def test_ugraph_init():
    nodes = (1, 2, 3)
    edges = ((1, 3), (1, 2), (2, 1))
    A = None

    exp_nodes, exp_edges, exp_A = (
        OrderedSet((1, 2, 3)),
        {(1, 2), (1, 3)},
        csr_matrix(
            ((1, 1, 1, 1), ((0, 1, 0, 2), (1, 0, 2, 0))), shape=(3, 3), dtype=bool
        ),
    )

    G = UGraph(nodes, edges, A)
    assert G.nodes == exp_nodes
    assert G.edges == exp_edges
    assert (G.A[G.A > 0].A == exp_A[exp_A > 0].A).all()


def test_tree_init():
    parents_init = (None, {2: 1, 3: 2, 4: 2})
    children_init = (None, {1: [2], 2: [3, 4]}, {1: [3]})
    edges_init = (None, ((1, 4), (2, 1)))

    expected = (
        (dict(), dict(), set()),
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}),
        ({2: 1, 3: 2, 4: 2}, {1: {2}, 2: {3, 4}}, {(1, 2), (2, 3), (2, 4)}),
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}, UserWarning),
        ({3: 1}, {1: {3}}, {(1, 3)}),
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}, UserWarning),
        ({2: 1, 3: 2, 4: 2}, {1: {2}, 2: {3, 4}}, {(1, 2), (2, 3), (2, 4)}),
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}, UserWarning),
        ({2: 1, 3: 2, 4: 2}, {1: {2}, 2: {3, 4}}, {(1, 2), (2, 3), (2, 4)}),
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}, UserWarning),
        InvalidTree,
        ({4: 1, 1: 2}, {1: {4}, 2: {1}}, {(1, 4), (2, 1)}, UserWarning),
    )

    for exp, (parents, children, edges) in zip(
        expected, itertools.product(parents_init, children_init, edges_init)
    ):
        if isinstance(exp, tuple):
            if len(exp) == 3:
                exp_parents, exp_children, exp_edges = exp
                T = Tree(parents=parents, children=children, edges=edges)
            else:
                exp_parents, exp_children, exp_edges, warning = exp
                with warns(warning):
                    T = Tree(parents=parents, children=children, edges=edges)
            assert T.parents == exp_parents
            assert T.children == exp_children
            assert T.edges == exp_edges
        else:
            assert raises(exp, Tree, parents=parents, children=children, edges=edges)
