"""
Run controlled and uncontrolled case and compare results

"""

import networkx as nx
import numpy as np

from . import sandpile
from . import cost


def run_on_graph(G, iterations=1000,
                 epsilon_control=0.1, epsilon_damage=0.001,
                 random_load=True):
    """
    Run on the given graph ``G``

    """
    # Maximum capacity is a node's degree
    degrees = nx.degree(G)
    C = np.array([degrees[i] for i in G.nodes()])
    # C = np.array([degrees[k] for k in sorted(degrees.keys())])
    if random_load:
        p = None
    else:
        p = {0: 0.52998, 1: 0.35651, 2: 0.11351}
    G, L0 = sandpile.initialize_loads(G, p=p, C=C)
    return do_run(G, L0, C, iterations, epsilon_control, epsilon_damage)


def run_regular(degree=3, nodes=1000, iterations=1000,
                epsilon_control=0.1, epsilon_damage=0.001):
    """
    Run on a k-regular graph. Returns (uncontrolled, controlled, df, costs).

    """
    # Generate graph and set up parameters
    degree = int(degree)
    nodes = int(nodes)
    iterations = int(iterations)
    G = nx.random_regular_graph(degree, nodes)
    # p = {0: 0.11351, 1: 0.35651}  # Flipped p and passed C, so that run_regular matches how things are done in run_scalefree
    p = {0: 0.52998, 1: 0.35651, 2: 0.11351}
    # Max capacity is linked to the graph's degree
    C = (degree - 1) * np.ones(nodes)
    G, L0 = sandpile.initialize_loads(G, p=p, C=C)
    return do_run(G, L0, C, iterations, epsilon_control, epsilon_damage)


def run_scalefree(alpha=0.60, beta=0.35, gamma=0.05, delta_in=0.2, delta_out=0,
                  nodes=1000, iterations=1000,
                  epsilon_control=0.1, epsilon_damage=0.001,
                  random_load=True):
    """
    Run on a scale-free graph. Returns (uncontrolled, controlled, df, costs).

    If random_load is True, load is distributed randomly initially.

    """
    # Generate graph and set up parameters
    nodes = int(nodes)
    iterations = int(iterations)
    G = nx.scale_free_graph(nodes, alpha, beta, gamma, delta_in, delta_out)
    G = G.to_undirected()  # scale_free_graph() creates a digraph
    # Maximum capacity is a node's degree
    degrees = nx.degree(G)
    C = np.array([degrees[i] for i in G.nodes()])
    if random_load:
        p = None
    else:
        p = {0: 0.52998, 1: 0.35651, 2: 0.11351}
    G, L0 = sandpile.initialize_loads(G, p=p, C=C)
    return do_run(G, L0, C, iterations, epsilon_control, epsilon_damage)


def do_run(G, L0, C, iterations, epsilon_control, epsilon_damage,
           epsilon=0.05, track_nodes=False):
    """
    Returns (uncontrolled, controlled, df, costs)

    """
    # Set up models
    uncontrolled = sandpile.Cascade(G=G, L0=L0, C=C, epsilon=0.05)
    controlled = sandpile.ControlledCascade(G=G, L0=L0, C=C,
                                            epsilon_control=epsilon_control,
                                            epsilon_damage=epsilon_damage)

    # Run uncontrolled case at least 1000 times to get through
    # transient phase, and throw away the results
    if iterations > 1000:
        transient_skip_iterations = iterations
    else:
        transient_skip_iterations = 1000
    uncontrolled.run(transient_skip_iterations, track_nodes=track_nodes)
    uncontrolled.run(iterations, init_only=True, track_nodes=track_nodes)
    controlled.run(iterations, init_only=True, track_nodes=track_nodes)

    for i in range(iterations):
        uncontrolled._run(i)
        # Carry over uncontrolled load to controlled case
        controlled.L = uncontrolled.L.copy()
        controlled._run(i)

    uncontrolled.gather_results()
    controlled.gather_results()

    # Get probabilities
    df = sandpile.analyze_cascades(uncontrolled)
    df = df.rename(columns={'probability': 'uncontrolled'})
    df2 = sandpile.analyze_cascades(controlled)
    df['controlled'] = df2.probability

    costs = cost.calculate_cost(uncontrolled, controlled)

    return (uncontrolled, controlled, df, costs)
