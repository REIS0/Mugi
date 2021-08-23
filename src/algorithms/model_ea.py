from typing import Union

import numpy as np


class ModelEA:
    def __init__(self, target: np.array, iterations: int) -> None:
        """
        Receive the target waveform and the fitness value.
        """
        self._size = len(target)
        self._target = target
        self._iterations = iterations

    def _generate_population(self) -> np.array:
        raise NotImplementedError

    def _evaluate(self, population: np.array, target: np.array) -> list:
        raise NotImplementedError

    def _best_fit(self, array: list) -> Union[float, float]:
        raise NotImplementedError

    def _recombine(self, parent1: np.array, parent2: np.array) -> np.array:
        raise NotImplementedError

    def _mutate(self, population: np.array) -> None:
        raise NotImplementedError

    def set_target(self, target: np.array) -> None:
        self._target = target
        self._size = len(target)

    def set_iterations(self, iterations: int) -> None:
        self._iterations = iterations

    def run(self) -> Union[float, np.array]:
        """
        Run the algorithm. Return a tuple with the fitness and
        the data.
        """
        raise NotImplementedError
