from numpy.random import default_rng
import numpy as np
import soundfile as sf

# TODO: implementar usando waveform
ALPHA = 0.5
INDV_SIZE = 4092


def generate_population() -> np.array:
    pop_size = 10
    population = np.empty((pop_size, INDV_SIZE))
    for i in range(pop_size):
        indv = default_rng().uniform(-1.0, 1.0, INDV_SIZE)
        population[i] = indv
    return population


def evaluate(population: np.array, target: np.array) -> list:
    evaluation = []
    for array in population:
        fit = 0
        for j in range(INDV_SIZE):
            fit += target[j] - array[j]
        evaluation.append((array, abs(fit)))
    return evaluation


def sum_fit(array: list) -> float:
    total = 0
    for _, i in array:
        total += i
    return total


def best_fit(array: list) -> float:
    best = 99
    for _, i in array:
        if i < best:
            best = i
    return best


def recombine(parent1: np.array, parent2: np.array) -> np.array:
    pop_size = 10
    population = np.empty((pop_size, INDV_SIZE))
    for i in range(pop_size):
        p_perc = int(default_rng().uniform(0.0, 1.0) * INDV_SIZE)
        population[i] = np.concatenate((parent1[p_perc:], parent2[:p_perc]))
    return population


def mutate(population: np.array, alpha: float) -> None:
    if alpha > 1.0 or alpha < 0.0:
        raise ValueError("Alpha nao esta entre 1 e 0")
    for indv in population:
        mutation = default_rng().uniform(-1.0, 1.0, INDV_SIZE)
        for j in range(INDV_SIZE):
            mutated = indv[j] + alpha * mutation[j]
            if mutated > 1.0:
                indv[j] = 1.0
            elif mutated < -1.0:
                indv[j] = -1.0
            else:
                indv[j] = mutated


def main() -> None:
    #! mudar por waveform
    target = default_rng().uniform(-1.0, 1.0, INDV_SIZE)
    print(f"Target: {target}")

    population = generate_population()
    evaluation = evaluate(population, target)

    fit = sum_fit(evaluation)
    generation = 0
    print(f"Fitness: {fit}; Generation: {generation}")

    while fit > 1.0:
        parent1 = (0, 99)
        parent2 = (0, 99)
        for i in evaluation:
            if i[1] < parent1[1]:
                parent1 = i
            elif i[1] < parent2[1]:
                parent2 = i

        new_pop = recombine(parent1[0], parent2[0])
        mutate(new_pop, ALPHA)
        evaluation = evaluate(new_pop, target)
        # fit = sum_fit(evaluation)
        fit = best_fit(evaluation)

        generation += 1
        print(f"Fitness: {fit}; Generation: {generation}")


if __name__ == "__main__":
    main()
