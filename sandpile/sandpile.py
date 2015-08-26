"""
Sand pile model of failure tolerance

"""

from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete

from . import core


DTYPE_INT = np.int


def initialize_loads(G, p=None, C=None):
    """
    Add initial loads to the given graph ``G``. Each node receives
    an initial load based on the probabilities given in ``p``.

    If ``C`` is None, the probabilities in p are the probability of a
    node having a certain load.

    If ``C`` (an array of node capacities) is given, the probabilities
    in p are interpreted as difference from capacity,

    If the probabilities together are less than one, then an additional
    entry in the probabilities list is added as {max given value + 1:
    1 - sum of all probabilities}.

    For example, p={1: 0.5, 2: 0.25} implies that with a probability of
    0.5 a node gets 1 less load than its capacity, with a probability of
    0.25 2 less than capacity, and with an implied probability of 0.25
    3 less than capacity (assuming that C was also given).

    The minimum load is zero.

    If ``p`` is None, ``C`` must be given, and each node receives a
    random load ranging from 0 to its capacity.

    Returns the graph itself with initial loads as ``"load"`` property,
    and a numpy array of the loads.

    """
    n = len(G.nodes())
    assert n == len(C)
    if not p:
        assert C is not None  # C must have been given if p is None
        L0 = np.array([np.random.random_integers(low=0, high=c)
                       for c in C], dtype=DTYPE_INT)
    else:  # p is given
        # Initialize node state matrix
        x = sorted(p.keys())
        px = [p[i] for i in x]
        # Add an additional value: (highest key given in p) + 1
        # and its associated probability is whatever's "left over"
        # this won't have much of an effect if the px pretty much
        # add up to 1 anyway
        x.append(x[-1] + 1)
        px.append(1 - sum(px))
        values = (x, px)
        # Create a discrete distribution and draw L0 from it
        distribution = rv_discrete(name='p', values=values)
        L0 = distribution.rvs(size=n)
        # Case where C is given
        if C is not None:
            L0 = C - L0
            L0[L0 < 0] = 0  # Ensure minimum load is zero
    # Add properties to nodes in G
    nodes_list = G.nodes()
    for index, item in enumerate(L0):
        node_name = nodes_list[index]
        G.node[node_name]['load'] = item
    return (G, L0)


def analyze_cascades(model):
    """
    Returns a pandas dataframe with the size and probability of cascades

    """
    toppled = model.nodes_toppled
    size = range(int(toppled.max()) + 1)
    prob = []
    for i in size:
        prob.append((toppled == i).sum() / float(len(toppled)))
    return pd.DataFrame({'size': size, 'probability': prob}).set_index('size')


def create_sparse_adjacency_list(nodes, nodes_indices, adjacency):
    """
    Creates a sparse adjacency list from list of nodes, nodes_indices, which
    is a dict of nodes giving their position in the nodes list, and
    adjacency, which is a NetworkX adjacency dict_of_lists.

    It returns (A, A_ptr, A_len):

    A contains the adjacency data, A_ptr contains the starting index in A
    for each row, and A_len contains the length of each row.

    """
    data = []
    row_ptr = []
    row_len = []
    ptr = 0

    for n in nodes:
        row = adjacency[n]
        row_ptr.append(ptr)  # Pointer to beginning of row
        row_len.append(len(row))  # Length of row
        row_as_int_indices = [nodes_indices[i] for i in row]
        data.extend(row_as_int_indices)  # Row data
        ptr += len(row)

    A = np.array(data, dtype=DTYPE_INT)
    A_ptr = np.array(row_ptr, dtype=DTYPE_INT)
    A_len = np.array(row_len, dtype=DTYPE_INT)

    return A, A_ptr, A_len


def get_row(A, A_ptr, A_len, i):
    return A[A_ptr[i]:A_ptr[i] + A_len[i]]


class Cascade(object):
    """
    Uncontrolled cascading failure model

    Args:

    ``G``: a NetworkX graph, where each node must have a 'load'
    property defining its initial load and a 'capacity' property
    defining its capacity (unless L0 and/or C are given)

    ``A``: adjacency list as a 2-dimensional numpy array (if G is not given),
    where the first column indicates the number of entries in the row, and
    the remaining entries give the adjacent nodes

    ``L0``: an array giving the initial loads for each node (optional,
    if not given, loads are extracted from the 'load' property of nodes)

    ``C``: maximum capacity of nodes (load beyond which they topple).
    Can either be a single integer or an array with same length as L0.

    ``epsilon``: dissipation probability

    """
    def __init__(self, G=None, L0=None, C=None, epsilon=0.05):
        super(Cascade, self).__init__()

        # NB always using G.nodes() to access nodes so that the sort order
        # is the same for A, L0 and C
        self.G = G
        nodes = G.nodes()
        nodes_indices = {node: i for i, node in enumerate(nodes)}

        # Initialize A (adjacency list)
        # This is always determined from G (cannot be passed separately)
        a = nx.to_dict_of_lists(G)
        A, A_ptr, A_len = create_sparse_adjacency_list(nodes, nodes_indices, a)
        self.A = A
        self.A_ptr = A_ptr
        self.A_len = A_len
        self.n = len(nodes)  # self.n is number of nodes

        # Initialize L and L0 (initial load)
        if L0 is not None:  # L0 given, so use it rather than G
            assert len(L0) == self.n  # Fail if L0 doesn't match G
            self.L0 = np.array(L0, dtype=DTYPE_INT)
        else:  # L0 not given
            l = [G.node[i]['load'] for i in nodes]
            self.L0 = np.array(l, dtype=DTYPE_INT)
        self.L = self.L0.copy()

        # Initialize C and C0 (initial capacity)
        if C is not None:  # C given, so use it rather than G
            if isinstance(C, int):
                self.C0 = C * np.ones(self.n, dtype=DTYPE_INT)
            else:
                assert len(C) == self.n
                self.C0 = np.array(C, dtype=DTYPE_INT)
        else:  # C not given
            c = [G.node[i]['capacity'] for i in nodes]
            self.C0 = np.array(c, dtype=DTYPE_INT)
        self.C_max = int(self.C0.max())
        self.C = self.C0.copy()

        self.epsilon = epsilon

    def _run(self, iteration):
        self.iteration = iteration
        core.run_uncontrolled(self.L, self.C, self.nodes_toppled,
                              self.sand_removed, self.A, self.A_ptr, self.A_len,
                              self.results_c, self.results_t,
                              self.epsilon, self.n, iteration,
                              self.track_nodes)

    def run(self, ts, init_only=False, track_nodes=True):
        """Run ``ts`` timesteps"""
        timesteps = int(ts)
        self.iteration = 0
        self.sand_removed = np.zeros(timesteps, dtype=DTYPE_INT)
        self.nodes_toppled = np.zeros(timesteps, dtype=DTYPE_INT)
        self.nodes_destroyed = np.zeros(timesteps, dtype=DTYPE_INT)
        # Result columns: total amount of sand in system, and then columns
        # to count the number of nodes with a given amount of sand (of
        # course a node can only have less than the toppling_load)
        shape_results = (timesteps, self.C_max + 1)
        # results_c: number of nodes with a given load
        if track_nodes:
            self.results_c = np.zeros(shape_results, dtype=DTYPE_INT)
            self.track_nodes = 1
        else:
            # Create a dummy object to pass into Cython function,
            # but it's not actually used for anything
            self.results_c = np.zeros((1, 1), dtype=DTYPE_INT)
            self.track_nodes = 0
        # results_t: total load in network
        self.results_t = np.zeros(timesteps, dtype=DTYPE_INT)
        if not init_only:
            for i in range(timesteps):
                self._run(i)

    def gather_results(self):
        if self.track_nodes == 1:
            self.results = pd.DataFrame(self.results_c)
            self.results['total'] = self.results_t
        else:
            self.results = pd.DataFrame({'total': self.results_t})


class ControlledCascade(Cascade):
    """
    Controlled cascading failure model

    Args:

    ``epsilon_control``: probability of control dissipating one unit
    of load from an overloaded node

    ``epsilon_damange``: probability of node damage when overloaded

    """
    def __init__(self, G=None, L0=None, C=None, epsilon=0.05,
                 epsilon_control=0.1, epsilon_damage=0.01):
        super(ControlledCascade, self).__init__(G=G, L0=L0, C=C,
                                                epsilon=epsilon)
        self.epsilon1 = epsilon_control  # additional dissipation probability
        self.epsilon_damage = epsilon_damage  # damage probability

    def _run(self, iteration):
        self.iteration = iteration
        np.copyto(self.C, self.C0)
        core.run_controlled(self.L, self.C,
                            self.nodes_toppled, self.nodes_destroyed,
                            self.sand_removed, self.A, self.A_ptr, self.A_len,
                            self.results_c, self.results_t,
                            self.epsilon, self.n, iteration,
                            self.track_nodes,
                            self.epsilon1, self.epsilon_damage)
