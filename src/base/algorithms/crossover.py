from typing import Callable

import numpy as np
from numpy.random import default_rng


def simple_one_point(
    pop: np.ndarray, best: np.ndarray, alpha: float, window: Callable
) -> np.ndarray:
    """
    Generate a random 'k' value between 0 and invidividual's size,
    the elements from 'k' to the end will be used in the crossover.
    """
    offspring = np.empty(pop.shape)
    for i in range(pop.shape[0]):
        offspring[i] = np.copy(best)
        k = default_rng().integers(0, best.shape[0])
        w = window(best[k:].shape[0])
        offspring[i][k:] = (w * best[k:]) * alpha + pop[i][k:] * (1 - alpha)
        offspring[i] = np.where(offspring[i] <= 1, offspring[i], 1.0)
        offspring[i] = np.where(offspring[i] >= -1, offspring[i], -1.0)
    return offspring


def simple_two_point(
    pop: np.ndarray, best: np.ndarray, alpha: float, window: Callable
) -> np.ndarray:
    """
    Generate a random 'k1' value between 0 and individual's size,
    then generate a random 'k2' value between 'k1' and individual's
    size, the values in the range 'k1' to 'k2' will be used in
    crossover operation.
    """
    offspring = np.empty(pop.shape)
    for i in range(pop.shape[0]):
        offspring[i] = np.copy(best)
        k1 = default_rng().integers(0, best.shape[0])
        k2 = default_rng().integers(k1, best.shape[0])
        w = window(best[k1 : k2 + 1].shape[0])
        offspring[i][k1 : k2 + 1] = (w * best[k1 : k2 + 1]) * alpha + pop[i][
            k1 : k2 + 1
        ] * (1 - alpha)
        offspring[i] = np.where(offspring[i] <= 1, offspring[i], 1.0)
        offspring[i] = np.where(offspring[i] >= -1, offspring[i], -1.0)
    return offspring


def single_arithmetic(
    pop: np.ndarray, best: np.ndarray, alpha: float, *_
) -> np.ndarray:
    """
    Generate a random 'k' value between 0 and individual size,
    the element in 'k' will be used in the crossover.
    """
    offspring = np.empty(pop.shape)
    for i in range(pop.shape[0]):
        offspring[i] = np.copy(best)
        k = default_rng().integers(0, best.shape[0])
        offspring[i][k] = best[k] * alpha + pop[i][k] * (1 - alpha)
        offspring[i] = np.where(offspring[i] <= 1, offspring[i], 1.0)
        offspring[i] = np.where(offspring[i] >= -1, offspring[i], -1.0)
    return offspring
