import numpy as np
from numpy.random import default_rng


def uniform_mutation(pop: np.ndarray, m_ind: float) -> np.ndarray:
    """
    A Beta array is generated with the same shape as individual
    together with a new completely random waveform. Then this
    waveform is mutiplied by Beta and then multiplied again by
    mutation amount. The resulting value is summed to the
    individual.
    """
    mutant = np.empty(pop.shape)
    for i in range(pop.shape[0]):
        beta = default_rng().random(size=pop.shape[1])
        mut = default_rng().uniform(-1.0, 1.0, pop.shape[1])
        mutant[i] = pop[i] + (mut * beta) * m_ind
        mutant[i] = np.where(mutant[i] <= 1, mutant[i], 1.0)
        mutant[i] = np.where(mutant[i] >= -1, mutant[i], -1.0)
    return mutant


def shuffle_indexes(pop: np.ndarray, m_ind: float) -> np.ndarray:
    """
    Shuffle index of array randomly according to mutation
    probability.
    """
    mutants = np.copy(pop)
    for mutant in mutants:
        for e in range(mutants.shape[0]):
            prob = default_rng().random()
            if prob <= m_ind:
                index = default_rng().integers(0, mutants.shape[0])
                mutant[e], mutant[index] = mutant[index], mutant[e]
    return mutants
