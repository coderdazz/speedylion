"""
Module contains functions related to simulations such as random portfolios
"""

import numpy as np


def simulatePortfolios(n, width, bound=None):
    """
    Generating random portfolios, sampling using Dirichlet distributions
    :param n: Number of random portfolios
    :param width: Number of assets
    :param bound: list, the weight bounds for individual instruments e.g. (0.03, 0.25) for 3% to 25%
    :return: Numpy array of n random portfolios
    """

    if bound is None:
        bound = (0.03, 0.25)

    alpha = np.ones(width)
    random_portfolios = np.random.dirichlet(alpha, n)
    lower = bound[0]
    upper = bound[1]

    min_array = np.full([n, width], lower)
    max_array = np.full([n, width], upper)

    # temporarily setting the maximum and minimum weights of random portfolios generated
    global_min = 0.0
    global_max = 0.5
    iteration = 1

    while global_min < lower - 0.003 or global_max > upper + 0.003 or iteration == 1:
        # find the amount of adjustments to fit lower bound
        min_exceeds = min_array - random_portfolios
        min_exceeds = np.where(min_exceeds > 0.00, min_exceeds, 0)

        # find the amount of adjustments to fit upper bound
        max_exceeds = max_array - random_portfolios
        max_exceeds = np.where(max_exceeds < 0.00, max_exceeds, 0)

        # negative exceeds means amount to be taken away and positive exceeds means amount to be given
        distributions = min_exceeds + max_exceeds
        no_of_distributions = np.where(distributions != 0.0, 0, 1).sum(axis=1, keepdims=True)
        remainder_distributions = - distributions.sum(axis=1, keepdims=True) / no_of_distributions
        distributions = np.where(distributions != 0.0, distributions, remainder_distributions)
        random_portfolios += distributions

        # justify the portfolios even though it's not needed
        random_portfolios /= random_portfolios.sum(axis=1, keepdims=True)
        global_min = random_portfolios.min()
        global_max = random_portfolios.max()

        iteration += 1

    print(n, ' random portfolios generated!')
    return random_portfolios

