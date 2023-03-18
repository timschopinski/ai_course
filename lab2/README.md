# Solving Knapsack Problem using genetic algorithm
The code implements a genetic algorithm that solves the Knapsack problem. The goal of the algorithm is to select a subset of items from a given set that maximizes the total value of the selected items while not exceeding a maximum weight limit (the capacity of the knapsack).
###

### Here is a brief explanation of the code:

- The get_big() function loads a set of items (represented as a Pandas dataframe) and the maximum weight limit of the knapsack.
- The initial_population() function generates an initial population of population_size individuals, where each individual is a binary string of length individual_size that encodes which items are selected (1) and which are not (0).
- The fitness() function calculates the fitness of an individual, which is the sum of the values of the selected items if the total weight of the selected items does not exceed the maximum weight limit. Otherwise, the fitness is 0.
- The population_best() function finds the best individual (i.e., the one with the highest fitness) in a given population.
- The main part of the code is a loop that runs for generations iterations. In each iteration, the current population is stored in population_history, and the genetic algorithm is applied to generate the next generation of individuals. The best individual and its fitness are stored in best_solution and best_fitness, respectively, and the history of the best fitness values is stored in best_history.
- At the end of the loop, the best solution and its fitness are printed, along with the total running time.
