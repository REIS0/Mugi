from typing import Callable

import numpy as np
import numpy.linalg as LA
from numpy.random import default_rng


class EA:
    def __init__(
        self,
        target: np.ndarray,
        pop_size: int,
        gen: int,
        n_rand: int = 1,
        alpha: float = 0.5,
        m_amount: float = 0.05,
        window: str = "hamming",
        distance: str = "euclidean",
    ) -> None:
        """
        Create a evolutionary class.

        Parameters
        ----------

        target : Numpy array.
                Waveform used to evaluate new waveforms.

        pop_size: Integer
                Population size.

        gen :   Integer.
                Number of generations to execute.

        n_rand : Integer. Optional.
                Value used in the selection, this sets to choose
                between 'n' random best waveforms in the generation,
                default value is '1'.

        alpha: Float. Optional.
                The amount used in crossover, must be between '0'
                and '1', default is '0.5'.

        m_amount: Float. Optional.
                Amount used in mutation operation, must be between
                '0' and '1', default is '0.05'.

        window: String. Optional.
                Window function used in crossover algorithm, can be
                only 'hamming', 'hanning' and 'blackman', default
                is 'hamming'.

        distance: String. Optional.
                Vector distance operation to evaluate the waveforms,
                can be 'euclidean', 'manhattan' and 'chebyshev',
                default is 'euclidean'.

        """
        self.__pop_size = pop_size
        self.__target = target
        self.__size = target.shape[0]
        self.__gen = gen
        self.set_rand_selection_number(n_rand)
        self.set_alpha_crossover(alpha)
        self.set_mut_amount(m_amount)
        self.set_window_function(window)
        self.set_distance_type(distance)

    def __generate_population(self) -> np.ndarray:
        pop = np.empty((self.__pop_size, self.__size))
        for i in range(self.__pop_size):
            pop[i] = default_rng().uniform(-1.0, 1.0, self.__size)
        return pop

    def __evaluate(self, pop: np.ndarray) -> np.ndarray:
        fits = np.empty(self.__pop_size)
        for i in range(self.__pop_size):
            fits[i] = LA.norm(self.__target - pop[i], ord=self.__norm)
        return fits

    def set_crossover(self, cross_alg: Callable) -> None:
        """Set the crossover algorithm."""
        self.__crossover = cross_alg

    def set_mutation(self, mut_alg: Callable) -> None:
        """Set the mutation algorithm."""
        self.__mutation = mut_alg

    def set_selection(self, sel_alg: Callable) -> None:
        """Set the selection algorithm."""
        self.__selection = sel_alg

    def set_distance_type(self, distance: str) -> None:
        """
        Set the distance type used to evaluate
        the waveforms, it can be 'euclidean',
        'manhattan' or 'chebyshev'.
        """
        if distance == "euclidean":
            self.__norm = 2
        elif distance == "manhattan":
            self.__norm = 1
        elif distance == "chebyshev":
            self.__norm = np.inf
        else:
            raise ValueError(f"Type '{distance}' is not valid.")

    def set_rand_selection_number(self, n: int) -> None:
        """
        Set the best 'n' individuals that can be
        selected.
        'n=1' is the same as just selecting
        the overall best from the population.
        If it's bigger than the population size, then
        will be equal to the population.
        """
        if n > self.__pop_size:
            self.__n_rand = self.__pop_size
        elif n <= 0:
            raise ValueError(f"Value '{n}' is less than 1.")
        else:
            self.__n_rand = n

    def set_alpha_crossover(self, a: float) -> None:
        """
        Set the alpha coefficiente used in
        crossover operations.
        It must be between 0 and 1.
        """
        if 0 <= a <= 1:
            self.__alpha = a
        else:
            raise ValueError(f"Value '{a}' out of range.")

    def set_mut_amount(self, mut: float) -> None:
        """
        Set the amount of mutation in
        the operation.
        It must be between 0 and 1.
        """
        if 0 <= mut <= 1:
            self.__mut_amount = mut
        else:
            raise ValueError(f"Value '{mut}' out of range.")

    def set_window_function(self, window: str) -> None:
        """
        Set the window function use in
        crossover, it can be 'hamming', 'hanning'
        or 'blackman'.
        """
        if window == "hamming":
            self.__window = np.hamming
        elif window == "hanning":
            self.__window = np.hanning
        elif window == "blackman":
            self.__window = np.blackman
        else:
            raise ValueError(f"Invalid '{window}' window option.")

    def run(self, verbose: bool = False) -> tuple[np.ndarray, dict]:
        """
        Execute the algorithm, set 'verbose' to true to
        print the information from each generation.
        """

        pop = self.__generate_population()
        fits = self.__evaluate(pop)
        best = self.__selection(pop, fits, self.__n_rand)

        log = {"gen": [], "std": [], "mean": [], "min": [], "max": []}

        if verbose:
            print("gen\t\tstd\t\tmean\t\tmin\t\tmax\n")

        for gen in range(self.__gen):

            offspring = self.__crossover(pop, best, self.__alpha, self.__window)
            mutant = self.__mutation(offspring, self.__mut_amount)
            pop = mutant
            fits = self.__evaluate(pop)
            best = self.__selection(pop, fits, self.__n_rand)

            std = np.std(fits)
            mean = np.mean(fits)
            mini = np.min(fits)
            maxi = np.amax(fits)
            log["gen"].append(gen + 1)
            log["std"].append(std)
            log["mean"].append(mean)
            log["min"].append(mini)
            log["max"].append(maxi)
            if verbose:
                print(f"{gen+1}\t\t{std:.4f}\t\t{mean:.4f}\t\t{mini:.4f}\t\t{maxi:.4f}")

        return best, log
