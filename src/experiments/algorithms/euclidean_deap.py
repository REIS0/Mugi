import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools
# !! temp
from icecream import ic
from numpy.random import default_rng

# !! does not work


class EuclideanDeap:
    def __init__(
        self,
        target: np.ndarray,
        pop_size: int,
        cxpb: float,
        ngen: int,
        fit=None,
        stats=None,
    ):
        "NOTICE: this algorithm does not work, use EuclideanEA instead."
        self.__target = target
        self.__pop_size = pop_size
        self.__ngen = ngen
        self.__size = target.shape[0]
        self.__cxpb = cxpb
        self.__stats = stats
        self.__fit = fit

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        self.__toolbox = base.Toolbox()
        self.__toolbox.register("rand_waveform", default_rng().uniform, -1.0, 1.0)
        # create individual with the same size as target
        self.__toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.__toolbox.rand_waveform,
            self.__size,
        )
        self.__toolbox.register(
            "population", tools.initRepeat, list, self.__toolbox.individual
        )

        self.__toolbox.register("evaluate", self.__eucl_eval)
        self.__toolbox.register("mate", self.__simpl_arith_recb)
        self.__toolbox.register("mutate", self.__uniform_mutate)
        # self.__toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.__toolbox.register("select", tools.selBest)

    def __eucl_eval(
        self, individual: np.ndarray
    ) -> tuple[int,]:
        """
        Calculate the euclidean distance between
        the target and generated waveform.
        """
        fit = LA.norm(self.__target - individual)
        return (fit,)

    def __simpl_arith_recb(
        self, best: np.ndarray, individual: np.ndarray
    ) -> tuple[np.ndarray,]:
        """
        Apply a simple arithmetic recombination
        with a individual and the best from the
        generation.
        """
        # !! Fix retorno com o deap
        k1 = default_rng().integers(0, self.__size)
        k2 = default_rng().integers(k1, self.__size) + 1
        ic(k1)
        ic(k2)
        ic(self.__cxpb)
        new_ind = self.__toolbox.clone(best)
        ic(new_ind)
        # TODO: arrumar o "+1"
        new_ind[k1:k2] = (self.__cxpb * individual[k1:k2]) + (
            (1 - self.__cxpb) * best[k1:k2]
        )
        ic(new_ind)
        ic(individual)
        individual = new_ind
        ic(individual)
        return (individual,)

    def __uniform_mutate(
        self, individual: np.ndarray, b: float
    ) -> tuple[np.ndarray,]:
        # TODO: editar range para 1 - b ate 1 + b
        beta = default_rng().uniform(1 - b, 1 + b, self.__size)
        ic(individual)
        individual = individual * beta
        # Maintain the original range
        # ** elementos em que a condicao for falsa serao trocados **
        ic(individual)
        np.where(individual < 1, individual, 1)
        np.where(individual > -1, individual, -1)
        ic(individual)
        return (individual,)

    def run(self, verbose=False):

        pop = self.__toolbox.population(n=self.__pop_size)

        fitnesses = list(map(self.__toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in pop]

        gen = 0

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (
            self.__stats.fields if self.__stats else []
        )

        record = self.__stats.compile(pop) if self.__stats else {}
        logbook.record(gen=gen, nevals=self.__pop_size, **record)

        # best ind in population
        best = pop[fits.index(np.min(fits))]
        # best = self.__toolbox.select(pop, 1)[0]
        # best = None
        ic("gen 0")
        ic(pop[0])
        ic(pop[1])

        if verbose:
            print("gen\t\tstd\t\tmean\t\tmin\t\tmax\n")

        while gen < self.__ngen:
            gen += 1

            # select the best individual
            offspring = self.__toolbox.select(pop, self.__pop_size)

            offspring = list(map(self.__toolbox.clone, pop))

            # crossover
            for child in offspring:
                # !! por algum motivo populacao nao e atualizada
                ic("--------------------------------------------")
                ic(child)
                self.__toolbox.mate(best, child)
                ic(child)
                del child.fitness.values

            # mutation
            b = default_rng().random()
            for mutant in offspring:
                # !! por algum motivo populacao nao e atualizada
                ic("--------------------------------------------")
                ic(mutant)
                self.__toolbox.mutate(mutant, b)
                ic(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.values]

            fitnesses = map(self.__toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = offspring

            fits = [ind.fitness.values[0] for ind in pop]

            best = pop[fits.index(np.min(fits))]

            record = self.__stats.compile(pop) if self.__stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # ic(pop[0])
            # ic(pop[1])

            if verbose:
                # print(logbook.stream)
                print(
                    f"""{gen}\t\t
                    {np.std(fits):.4f}
                    \t\t{np.mean(fits):.4f}
                    \t\t{np.min(fits):.4f}
                    \t\t{np.amax(fits):.4f}"""
                )

            if self.__fit and min(fits) < self.__fit:
                break

        return best, logbook
