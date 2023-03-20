import random
import numpy as np


def roulette_wheel_selection(population, fitness_fn, n_selection):
    fitnesses = [fitness_fn(individual) for individual in population]
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    cumulative_probabilities = list(np.cumsum(probabilities))
    selected_individuals = []
    for i in range(n_selection):
        r = random.uniform(0, 1)
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                selected_individuals.append(population[j])
                break
    return selected_individuals
