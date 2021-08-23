from typing import Union

import numpy as np


class ModelEA:
    def __init__(self, target: np.array, fit_thresh: float) -> None:
        """
        Receive the target waveform and the fitness value.
        """
        raise NotImplemented

    def __generate_population(self) -> np.array:
        raise NotImplemented

    def __evaluate(self, population: np.array, target: np.array) -> list:
        raise NotImplemented

    def __best_fit(self, array: list) -> Union[float, float]:
        raise NotImplemented

    def __recombine(self, parent1: np.array, parent2: np.array) -> np.array:
        raise NotImplemented

    def __mutate(self, population: np.array) -> None:
        raise NotImplemented

    def set_target(self, target: np.array) -> None:
        self.__target = target

    def run(self) -> Union[float, np.array]:
        """
        Run the algorithm. Return a tuple with the fitness and
        the data.
        """
        raise NotImplemented
