# distutils: language = c++

from __future__ import division

import cython
import numpy as np
import time
cimport cython
from cython cimport view
cimport numpy as np
from libc.stdlib cimport malloc, free, rand, srand
from libc.stdio cimport printf
from libc.math cimport floor

cdef extern from "stdlib.h":
    int RAND_MAX
cdef double RAND_MAX_F = float(RAND_MAX)
# Random seed with current UNIX timestamp
srand(int(time.time()))

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

# C++ imports
from libcpp.deque cimport deque


@cython.cdivision(True)
cdef int randint(int n) nogil:
    """Returns a random int from 0 to n"""
    return rand() % n


@cython.cdivision(True)
cdef double randfloat() nogil:
    """Returns a random float in range (0, 1)"""
    return rand() / RAND_MAX_F


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int random_sample(int *source, int source_len, int n) nogil:
    """
    Randomly samples ``n`` elements from the ``source`` array,
    filling them back into the ``source`` array (which can then be
    treated as having length ``n``).

    Returns 0.

    """
    cdef:
        int i = 0
        int removed = 0
        int start = randint(source_len)
        int *temp = <int *>malloc(n * sizeof(int))

    # Make the random selection, storing it in the temp array
    while n > 0:
        # Probability of selection = number needed / number left
        # Carries small extra cost because of Python zero-division checks
        if randfloat() > (n / source_len - removed):
            # Source: http://stackoverflow.com/a/48089/397746
            temp[n - 1] = source[i]
            removed += 1
            n -= 1
        i += 1

    # Fill selected values in temp back into source array
    for i in range(n):
        source[i] = temp[i]
    free(temp)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int update_neighbors(DTYPE_INT_t[:] L,
                          int *neighbors,
                          int n_neighbors,
                          int receiver,
                          double epsilon,
                          DTYPE_INT_t[:] sand_removed,
                          int iteration) nogil:
    """
    Dissipate to neighbors

    Updates L and neighbors in-place.

    Returns updated n_neighbors (changed if random selection was used).

    """
    cdef:
        int i
        int j
        int load = L[receiver]
        int per_neighbor
        int count_leftover

    if n_neighbors == load:
        per_neighbor = 1
    elif n_neighbors > load:
        if load == 0:
            per_neighbor = 0
            n_neighbors = 0
        else:
            per_neighbor = 1
            # random_sample modifies the passed neighbors array
            random_sample(neighbors, n_neighbors, load)
            n_neighbors = load
    elif n_neighbors == 0:
        sand_removed[iteration] += load
    else:  # n_neighbors < load
        # Some neighbors get more sand, as determined by count_leftover
        per_neighbor = <int>floor(load / n_neighbors)
        count_leftover = load - (per_neighbor * n_neighbors)
        # No random sampling, simply fill up to count_leftover
        for i in range(count_leftover):
            if randfloat() > epsilon:
                L[neighbors[i]] += 1
    # For each (chosen) neighbor, add per_neighbor of load
    if n_neighbors > 0:
        for i in range(n_neighbors):
            for j in range(per_neighbor):
                if randfloat() > epsilon:
                    L[neighbors[i]] += 1
    L[receiver] = 0

    return n_neighbors


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int topple(DTYPE_INT_t[:] L,
                DTYPE_INT_t[:] nodes_toppled,
                DTYPE_INT_t[:] sand_removed,
                double epsilon,
                int iteration,
                int *neighbors,
                int n_neighbors,
                int receiver,
                deque[int] *fifo):
    """
    Topple a node and distribute its load to its neighbors.

    Returns updated n_neighbors.

    """

    nodes_toppled[iteration] += 1
    n_neighbors = update_neighbors(L, neighbors, n_neighbors,
                                   receiver, epsilon,
                                   sand_removed, iteration)
    if n_neighbors > 0:
        for i in range(n_neighbors):
            fifo.push_back(neighbors[i])

    return n_neighbors


@cython.boundscheck(False)
@cython.wraparound(False)
def run_uncontrolled(DTYPE_INT_t[:] L,
                     DTYPE_INT_t[:] C,
                     DTYPE_INT_t[:] nodes_toppled,
                     DTYPE_INT_t[:] sand_removed,
                     DTYPE_INT_t[:] A,
                     DTYPE_INT_t[:] A_ptr,
                     DTYPE_INT_t[:] A_len,
                     DTYPE_INT_t[:, :] results_c,
                     DTYPE_INT_t[:] results_t,
                     double epsilon,
                     int n,
                     int iteration,
                     int track_nodes):
    """
    Runs one uncontrolled iteration

    L, C, nodes_toppled, sand_removed, results_c and results_t
    are modified in-place.

    """
    cdef:
        int i
        int receiver
        int root = randint(n)
        int *neighbors
        int n_neighbors
        deque[int] *fifo = new deque[int]()

    L[root] += 1  # Add sand to root
    fifo.push_back(root)  # Add root to queue
    while fifo.size() > 0:
        receiver = fifo.front()  # Access first element
        fifo.pop_front()  # Delete first element after accessing it
        if L[receiver] > C[receiver]:
            n_neighbors = A_len[receiver]
            neighbors = <int *>malloc(n_neighbors * sizeof(int))
            for i in range(n_neighbors):
                neighbors[i] = A[A_ptr[receiver] + i]
            # Topple the node and distribute load to neighbors
            topple(L, nodes_toppled, sand_removed, epsilon, iteration,
                   neighbors, n_neighbors, receiver, fifo)
            free(neighbors)
    del fifo

    # Gather results
    for i in range(n):
        if track_nodes == 1:
            results_c[iteration, L[i]] += 1
        results_t[iteration] += L[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def run_controlled(DTYPE_INT_t[:] L,
                   DTYPE_INT_t[:] C,
                   DTYPE_INT_t[:] nodes_toppled,
                   DTYPE_INT_t[:] nodes_destroyed,
                   DTYPE_INT_t[:] sand_removed,
                   DTYPE_INT_t[:] A,
                   DTYPE_INT_t[:] A_ptr,
                   DTYPE_INT_t[:] A_len,
                   DTYPE_INT_t[:, :] results_c,
                   DTYPE_INT_t[:] results_t,
                   double epsilon,
                   int n,
                   int iteration,
                   int track_nodes,
                   double epsilon1,
                   double epsilon_damage):
    """
    Runs one controlled iteration

    L, C, nodes_toppled, nodes_destroyed, sand_removed, results_c
    and results_t are modified in-place.

    """
    cdef:
        int i
        int ni
        int receiver
        int dissipated
        int root = randint(n)
        int *neighbors
        int n_neighbors
        deque[int] *fifo = new deque[int]()

    L[root] += 1  # Add sand to root
    fifo.push_back(root)  # Add root to queue
    while fifo.size() > 0:
        receiver = fifo.front()  # Access first element
        fifo.pop_front()  # Delete first element after accessing it
        if L[receiver] > C[receiver]:
            ##
            # Get neighbors
            ##
            n_neighbors = A_len[receiver]
            neighbors = <int *>malloc(n_neighbors * sizeof(int))
            # For each adjacent neighbor, check if destroyed (C <= 0),
            # and only add to neighbors if it is not.
            ni = 0
            for i in range(n_neighbors):
                if C[A[A_ptr[receiver] + i]] > 0:
                    neighbors[ni] = A[A_ptr[receiver] + i]
                    ni += 1
                else:
                    n_neighbors -= 1

            ##
            # Control part 1: damage
            ##
            if randfloat() < epsilon_damage:
                # First, we topple and distribute load to neighbors
                # NB this counts towards nodes_toppled inside topple()
                # as well as towards nodes_destroyed just below
                n_neighbors = topple(L, nodes_toppled, sand_removed, epsilon,
                                     iteration, neighbors, n_neighbors,
                                     receiver, fifo)
                # Record that receiver is destroyed, its capacity becomes 0.
                C[receiver] = 0
                nodes_destroyed[iteration] += 1
                # Neighbors are damaged by reduction of capacity
                # This may result in already destroyed neighbors having
                # capacities <0
                for i in range(n_neighbors):
                    C[neighbors[i]] -= 1
            else:
                ##
                # Control part 2: dissipation from overloaded node
                ##
                dissipated = 0
                for i in range(L[receiver]):
                    if randfloat() < epsilon1:
                        dissipated += 1
                L[receiver] -= dissipated
                sand_removed[iteration] += dissipated
                if L[receiver] > C[receiver]:
                    ##
                    # Control part 3: after dissipation,
                    # check if node is still above capacity, and if so,
                    # topple the node
                    ##
                    topple(L, nodes_toppled, sand_removed, epsilon, iteration,
                           neighbors, n_neighbors, receiver, fifo)
            free(neighbors)
    del fifo

    # Gather results
    for i in range(n):
        if track_nodes == 1:
            results_c[iteration, L[i]] += 1
        results_t[iteration] += L[i]
