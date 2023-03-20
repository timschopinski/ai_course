from itertools import compress
import random
import time
import matplotlib.pyplot as plt
from fitness import fitness
from data import *
from lab2.crossover import single_point_crossover
from lab2.mutation import mutate
from lab2.selection import roulette_wheel_selection


def initial_population(individual_size, population_size):
    return [
        [random.choice([True, False]) for _ in range(individual_size)]
        for _ in range(population_size)
    ]


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


if __name__ == "__main__":
    items, knapsack_max_capacity = get_big()
    print(items)

    population_size = 100
    generations = 200
    n_selection = 20
    n_elite = 1
    mutation_rate = 0.001
    start_time = time.time()
    best_solution = None
    best_fitness = 0
    population_history = []
    best_history = []
    population = initial_population(len(items), population_size)

    for _ in range(generations):
        population_history.append(population)

        # selection
        selected_parents = roulette_wheel_selection(
            population,
            lambda ind: fitness(items, knapsack_max_capacity, ind),
            n_selection,
        )

        # crossover
        offspring = single_point_crossover(selected_parents)

        # mutation
        offspring = mutate(offspring)

        # elitism
        elite_individual, elite_fitness = population_best(
            items, knapsack_max_capacity, population
        )
        elite_offspring = [elite_individual] * n_elite
        offspring += elite_offspring

        # new population
        population = offspring

        best_individual, best_individual_fitness = population_best(
            items, knapsack_max_capacity, population
        )
        if best_individual_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = best_individual_fitness
        best_history.append(best_fitness)

    end_time = time.time()
    total_time = end_time - start_time
    print("Best solution:", list(compress(items["Name"], best_solution)))
    print("Best solution value:", best_fitness)
    print("Time: ", total_time)

    # plot generations
    x = []
    y = []
    top_best = 10
    for i, population in enumerate(population_history):
        plotted_individuals = min(len(population), top_best)
        x.extend([i] * plotted_individuals)
        population_fitnesses = [
            fitness(items, knapsack_max_capacity, individual)
            for individual in population
        ]
        population_fitnesses.sort(reverse=True)
        y.extend(population_fitnesses[:plotted_individuals])
    plt.scatter(x, y, marker=".")
    plt.plot(best_history, "r")
    plt.scatter(100, best_fitness, marker="o", color='green')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
