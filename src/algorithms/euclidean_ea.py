from typing import Union

import numpy as np
import numpy.linalg as LA
from numpy.random import default_rng

from src.algorithms.model_ea import ModelEA

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
        population = np.empty((self._pop_size, self._size), dtype=np.float_)
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
        evaluation = np.empty(self._pop_size, dtype=np.float_)
        for a in range(self._pop_size):
            evaluation[a] = LA.norm(self._target - population[a])
        return evaluation

    def _best_fit(self, array: np.ndarray) -> Union[np.float_, np.float_]:
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
        new_population = np.empty((self._pop_size, self._size), dtype=np.float_)
        for i in range(self._pop_size):
            # Simple Arithmetic Recombination
            new_population[i] = population[best]
            k1 = default_rng().integers(
                0,
                self._size,
            )
            k2 = default_rng().integers(k1, self._size)
            for j in range(k1, k2 + 1):
                new_population[i][j] = (
                    ALPHA * population[i][j] + (1 - ALPHA) * new_population[i][j]
                )
        return new_population

    def _mutate(self, population: np.ndarray) -> None:
        """
        Mutation occur by generating a random value "prob" between 0 and 1.
        The "beta" is made by a random value between 1 and 1 minus "prob".
        """
        # Uniform mutation
        prob = default_rng().random(self._size)
        for i in range(self._pop_size):
            beta = default_rng().uniform(1 - prob, 1)
            noise = default_rng().uniform(-1.0, 1.0, self._size)
            population[i] += beta * noise
            population[i] = np.where(population[i] > 1, population[i], 1)
            population[i] = np.where(population[i] < -1, population[i], -1)

    def run(self) -> Union[np.float_, np.ndarray]:
        gen = 1
        population = self._generate_population()
        evaluation = self._evaluate(population)

        fit, index = self._best_fit(evaluation)

        best_fit = (fit, population[index])

        while True:
            gen += 1
            if self._iterations and gen >= self._iterations:
                break

            population = self._recombine(index, population)
            self._mutate(population)
            evaluation = self._evaluate(population)
            fit, index = self._best_fit(evaluation)

            if fit < best_fit[0]:
                best_fit = (fit, population[index])

            if self._fit and best_fit[0] <= self._fit:
                break

        return best_fit
