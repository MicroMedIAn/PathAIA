from pathaia.graphs import Graph, UGraph, Tree
from scipy.sparse import csr_matrix
from pytest import raises
import itertools
from ordered_set import OrderedSet


def test_graph_init():
    nodes_init = (None, (1, 2, 3), "123")
    edges_init = (None, ((1, 3), (1, 2), (2, 1)))
    A_init = (None, csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool))

    expected = (
        (OrderedSet(), [], csr_matrix((0, 0), dtype=bool)),
        (
            OrderedSet((0, 1, 2)),
            [(0, 1), (1, 2)],
            csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 3, 2)),
            [(1, 3), (1, 2), (2, 1)],
            csr_matrix(((1, 1, 1), ((0, 0, 2), (1, 2, 0))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 3, 2)),
            [(1, 3), (1, 2), (2, 1)],
            csr_matrix(((1, 1, 1), ((0, 0, 2), (1, 2, 0))), shape=(3, 3), dtype=bool),
        ),
        (OrderedSet((1, 2, 3)), [], csr_matrix((3, 3), dtype=bool)),
        (
            OrderedSet((1, 2, 3)),
            [(1, 2), (2, 3)],
            csr_matrix(((1, 1), ((0, 1), (1, 2))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 2, 3)),
            [(1, 3), (1, 2), (2, 1)],
            csr_matrix(((1, 1, 1), ((0, 0, 1), (2, 1, 0))), shape=(3, 3), dtype=bool),
        ),
        (
            OrderedSet((1, 2, 3)),
            [(1, 3), (1, 2), (2, 1)],
            csr_matrix(((1, 1, 1), ((0, 0, 1), (2, 1, 0))), shape=(3, 3), dtype=bool),
        ),
        (OrderedSet(("1", "2", "3")), [], csr_matrix((3, 3), dtype=bool)),
        (
            OrderedSet(("1", "2", "3")),
            [("1", "2"), ("2", "3")],
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
            assert (G.A[G.A > 0] == exp_A[exp_A > 0]).A.all()
        else:
            assert raises(exp, Graph, nodes, edges, A)
