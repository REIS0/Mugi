from typing import Union

import numpy as np
from numpy.random import default_rng

from src.experiments.algorithms.model_ea import ModelEA


class SimpleEA(ModelEA):
    def __init__(
        self,
        target: np.ndarray,
        pop_size: int,
        fit: float,
        alpha=0.5,
        iterations=None,
    ) -> None:
        """
        Fitness is calculated by the difference between the sum
        of the elements.
        """
        self.set_alpha(alpha)
        super().__init__(target, pop_size, fit, iterations=iterations)

    def set_alpha(self, alpha: float) -> None:
        """
        Set the alpha coefficiente for mutation, must be between 0 and 1.
        Default is 0.5.
        """
        if alpha > 1.0 or alpha < 0.0:
            raise ValueError("Alpha nao esta entre 1 e 0")
        self.__alpha = alpha

    def _generate_population(self) -> np.ndarray:
        population = np.empty((self._pop_size, self._size))
        for i in range(self._pop_size):
            indv = default_rng().uniform(-1.0, 1.0, self._size)
            population[i] = indv
        return population

    def _evaluate(self, population: np.ndarray) -> list:
        evaluation = []
        for array in population:
            fit = 0
            for j in range(self._size):
                fit += self._target[j] - array[j]
            evaluation.append((array, abs(fit)))
        return evaluation

    def _best_fit(self, array: list) -> Union[float, float]:
        best = 99
        index = None
        for i in range(len(array)):
            if array[i][1] < best:
                best = array[i][1]
                index = i
        return best, index

    def _recombine(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        pop_size = 10
        population = np.empty((pop_size, self._size))
        for i in range(pop_size):
            p_perc = int(default_rng().uniform(0.0, 1.0) * self._size)
            population[i] = np.concatenate(
                (parent1[p_perc:], parent2[:p_perc])
            )
        return population

    def _mutate(self, population: np.ndarray) -> None:
        for indv in population:
            mutation = default_rng().uniform(-1.0, 1.0, self._size)
            for j in range(self._size):
                mutated = indv[j] + self.__alpha * mutation[j]
                if mutated > 1.0:
                    indv[j] = 1.0
                elif mutated < -1.0:
                    indv[j] = -1.0
                else:
                    indv[j] = mutated

    def run(self) -> Union[float, np.ndarray]:
        gen = 1
        population = self._generate_population()
        evaluation = self._evaluate(population)

        fit, index = self._best_fit(evaluation)

        best_fit = (fit, evaluation[index][0])

        while True:
            gen += 1
            if self._iterations and gen > self._iterations:
                break
            parent1 = (0, 99)
            parent2 = (0, 99)
            for i in evaluation:
                if i[1] < parent1[1]:
                    parent1 = i
                elif i[1] < parent2[1]:
                    parent2 = i

            new_pop = self._recombine(parent1[0], parent2[0])
            self._mutate(new_pop)
            evaluation = self._evaluate(new_pop)
            fit, index = self._best_fit(evaluation)

            if fit < best_fit[0]:
                best_fit = (fit, evaluation[index][0])

            if self._fit and best_fit[0] <= self._fit:
                break

        return best_fit
