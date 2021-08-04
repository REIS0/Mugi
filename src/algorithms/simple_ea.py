from typing import Union
from numpy.random import default_rng
import numpy as np


class SimpleEA:
    def __init__(self, size: int, fit_thresh: float, alpha=0.5) -> None:
        self.set_alpha(alpha)
        self.__size = size
        self.__fit_thresh = fit_thresh

    def set_alpha(self, alpha: float) -> None:
        """
        Set the alpha coefficiente for mutation, must be between 0 and 1.
        Default is 0.5.
        """
        if alpha > 1.0 or alpha < 0.0:
            raise ValueError("Alpha nao esta entre 1 e 0")
        self.__alpha = alpha

    def __generate_population(self) -> np.array:
        pop_size = 10
        population = np.empty((pop_size, self.__size))
        for i in range(pop_size):
            indv = default_rng().uniform(-1.0, 1.0, self.__size)
            population[i] = indv
        return population

    def __evaluate(self, population: np.array, target: np.array) -> list:
        evaluation = []
        for array in population:
            fit = 0
            for j in range(self.__size):
                fit += target[j] - array[j]
            evaluation.append((array, abs(fit)))
        return evaluation

    def __best_fit(self, array: list) -> Union[float, float]:
        best = 99
        index = None
        for i in range(len(array)):
            if array[i][1] < best:
                best = array[i][1]
                index = i
        return best, index

    def __recombine(self, parent1: np.array, parent2: np.array) -> np.array:
        pop_size = 10
        population = np.empty((pop_size, self.__size))
        for i in range(pop_size):
            p_perc = int(default_rng().uniform(0.0, 1.0) * self.__size)
            population[i] = np.concatenate((parent1[p_perc:], parent2[:p_perc]))
        return population

    def __mutate(self, population: np.array) -> None:
        for indv in population:
            mutation = default_rng().uniform(-1.0, 1.0, self.__size)
            for j in range(self.__size):
                mutated = indv[j] + self.__alpha * mutation[j]
                if mutated > 1.0:
                    indv[j] = 1.0
                elif mutated < -1.0:
                    indv[j] = -1.0
                else:
                    indv[j] = mutated

    def run(self, target: np.array, fit_type="best") -> None:
        """
        Run the algorithm.
        """
        population = self.__generate_population()
        evaluation = self.__evaluate(population, target)

        fit, index = self.__best_fit(evaluation)
        generation = 0
        print(f"Fitness: {fit}; Generation: {generation}")

        while fit > self.__fit_thresh:
            parent1 = (0, 99)
            parent2 = (0, 99)
            for i in evaluation:
                if i[1] < parent1[1]:
                    parent1 = i
                elif i[1] < parent2[1]:
                    parent2 = i

            new_pop = self.__recombine(parent1[0], parent2[0])
            self.__mutate(new_pop)
            evaluation = self.__evaluate(new_pop, target)
            fit, index = self.__best_fit(evaluation)

            generation += 1
            print(f"Fitness: {fit}; Index: {index}; Generation: {generation}")
