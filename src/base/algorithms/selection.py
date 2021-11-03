from random import randint

import numpy as np
from icecream import ic


def select_best(
    pop: np.ndarray, fits: np.ndarray, n_rand: int
) -> tuple[int, np.ndarray]:
    """
    Select randomly a individual from 'n' best fits, if
    'n' equals to 1, then return the best overall individual.
    """
    if n_rand == 1:
        best = np.min(fits)
        index = np.where(fits == best)[0][0]
    else:
        sorted_a = np.sort(fits)
        # Select random from n best elements.
        rng = (
            randint(0, n_rand - 1)
            if n_rand < fits.shape[0]
            else randint(0, fits.shape[0] - 1)
        )
        selected = sorted_a[rng]
        # ** Since numpy returns array with all positions the element occur,
        # ** a random one is selected, if there's only one elemenet, then
        # ** select it.
        best = np.where(fits == selected)[0]
        if len(best) == 1:
            index = best[0]
        else:
            index = randint(0, len(best) - 1)
    return pop[index]
