import numpy as np
import numpy.linalg as LA
from icecream import ic
from numpy.random import default_rng

from src.experiments.algorithms.model_ea import ModelEA

# TODO: alpha como parametro
ALPHA = 0.5


class EuclideanEA(ModelEA):
    def __init__(
        self, target: np.ndarray, pop_size: int, fit=None, iterations=None
    ) -> None:
        """
        Fitness is calculated by the euclidean distance.
        """
        super().__init__(target, pop_size, fit, iterations=iterations)

    def _generate_population(self) -> np.ndarray:
        population = np.empty((self._pop_size, self._size))
        for i in range(self._pop_size):
            indv = default_rng().uniform(-1.0, 1.0, self._size)
            population[i] = indv
        return population

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate the population by calculating the euclidean
        distance between each individual and target.
        Returns an array with the fitness according the index from the
        population.
        """
        evaluation = np.empty(self._pop_size)
        for a in range(self._pop_size):
            evaluation[a] = LA.norm(self._target - population[a])
        return evaluation

    def _best_fit(self, array: np.ndarray) -> tuple[np.float_, np.float_]:
        """
        Get the individual with the best fitness. Return the
        fitness value and index in the population.
        """
        best = np.min(array)
        index = np.where(array == best)[0][0]
        return best, index

    def _recombine(self, best: int, population: np.ndarray) -> np.ndarray:
        """
        Crossover to create a new population.
        "best" is the index for the best generated waveform,
        "population" is the population itself.
        """
        new_pop = np.empty((self._pop_size, self._size))
        for i in range(self._pop_size):
            # Simple Arithmetic Recombination
            new_ind = np.copy(population[best])
            k1 = default_rng().integers(
                0,
                self._size,
            )
            k2 = default_rng().integers(k1, self._size) + 1
            new_ind[k1 : k2 + 1] = (
                ALPHA
                * (np.hamming(population[i][k1 : k2 + 1].shape[0]) * population[i][k1 : k2 + 1])
            ) + ((1 - ALPHA) * population[best][k1 : k2 + 1])
            # ic(new_ind)
            new_pop[i] = new_ind
        return new_pop

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """
        Generate a random value "b" between 0 and 1, then generating a
        "beta" made by a random values between 1 minus "b" and 1 plus "b". Apply
        the noise multiplied by "beta" to the individual in the population.
        """
        # Uniform mutation
        b = default_rng().random()
        # Copy the population
        new = np.copy(population)
        for i in range(self._pop_size):
            # ic(i)
            # ic(new[i])
            beta = default_rng().uniform(1 - b, 1 + b, self._size)
            # ic(beta)
            new[i] = new[i] * beta
            # ic(new[i])
            # ** elementos em que a condicao for falsa serao trocados **
            new[i] = np.where(new[i] < 1, new[i], 1)
            # ic(new[i])
            new[i] = np.where(new[i] > -1, new[i], -1)
            # ic(new[i])
        return new

    def run(self, verbose=False) -> tuple[np.float_, np.ndarray]:
        gen = 0
        population = self._generate_population()
        evaluation = self._evaluate(population)

        fit, index = self._best_fit(evaluation)

        best_fit = (fit, population[index])

        if verbose:
            print("gen\t\tstd\t\tmean\t\tmin\t\tmax\n")

        while True:
            gen += 1
            if self._iterations and gen > self._iterations:
                break

            if fit < best_fit[0]:
                best_fit = (fit, population[index])

            if self._fit and best_fit[0] <= self._fit:
                break

            # ic(gen)
            # ic(population)
            # ic('crossover')
            population = self._recombine(index, population)
            # ic(population)
            # ic('mutacao')
            population = self._mutate(population)
            # ic(population)
            evaluation = self._evaluate(population)
            # ic(evaluation)
            fit, index = self._best_fit(evaluation)

            if verbose:
                print(
                    f"{gen}\t\t{np.std(evaluation):.4f}\t\t{np.mean(evaluation):.4f}\t\t{np.min(evaluation):.4f}\t\t{np.max(evaluation):.4f}"
                )

        return best_fit
