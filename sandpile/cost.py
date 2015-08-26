"""
Functionality to calculate cost of control

"""

from __future__ import division

import numpy as np


def calculate_cost(uncontrolled_model, controlled_model):
    # Cost and cost parameters hardcoded for now
    cost_removedsand = 0.6
    cost_damage = 1.0
    cost_no_cascade = -1.0
    c = 0.005
    alpha = 1.5

    # Get iteration count and make sure it matches for both models
    iterations = uncontrolled_model.iteration
    assert iterations == controlled_model.iteration

    # Cost containers
    cost_uc = []  # Lists of uncontrolled
    cost_c = []  # and controlled costs

    # Result containers
    cascades_uc = uncontrolled_model.nodes_toppled
    cascades_c = controlled_model.nodes_toppled
    damaged_c = controlled_model.nodes_destroyed
    sand_removed_c = controlled_model.sand_removed

    for i in range(iterations):
        c_uc = 0
        c_c = 0
        # Uncontrolled cost
        if cascades_uc[i] == 0:
            c_uc += cost_no_cascade
        else:
            c_uc += c * cascades_uc[i] ** alpha
        # Controlled cost
        if cascades_c[i] == 0:
            c_c += cost_no_cascade
        else:
            c_c += c * cascades_c[i] ** alpha
        c_c += (cost_removedsand * sand_removed_c[i]
                + cost_damage * damaged_c[i])
        # Accumulate costs
        cost_uc.append(c_uc)
        cost_c.append(c_c)

    return np.mean(cost_uc), np.mean(cost_c)
