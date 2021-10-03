from typing import Union

import numpy as np


class ModelEA:
    def __init__(
        self, target: np.ndarray, pop_size: int, fit=None, iterations=None
    ) -> None:
        """
        Receive the target waveform.
        Is needed to set a fitness value or a max iteration (which is equal
        to the generation) value.
        """
        self._size = len(target)
        self._target = target
        self._pop_size = pop_size
        if not fit and not iterations:
            raise TypeError(
                "Missing fitness and iterations value. Please assign one of them."
            )
        self._fit = fit
        self._iterations = iterations

    def _generate_population(self) -> np.ndarray:
        raise NotImplementedError

    def _evaluate(self, population: np.ndarray, target: np.ndarray) -> list:
        raise NotImplementedError

    def _best_fit(self, array: list) -> Union[float, float]:
        raise NotImplementedError

    def _recombine(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _mutate(self, population: np.ndarray) -> None:
        raise NotImplementedError

    def set_target(self, target: np.ndarray) -> None:
        self._target = target
        self._size = len(target)

    def set_iterations(self, iterations: int) -> None:
        self._iterations = iterations

    def set_fit(self, fit: float) -> None:
        self._fit = fit

    def run(self) -> Union[float, np.ndarray]:
        """
        Run the algorithm. Return a tuple with the fitness and
        the data.
        """
        raise NotImplementedError
