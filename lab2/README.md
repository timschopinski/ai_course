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

## The Genetic Algorithm Breakdown
- ### Selection
```python
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
```
This function implements the roulette wheel selection method for selecting individuals from a population for the next generation. The selection probability for each individual is proportional to its fitness score, so individuals with higher fitness scores have a higher probability of being selected.

#### Here is a step-by-step breakdown of the function:

- Calculate the fitness scores for each individual in the population using the fitness function provided as an argument.
- Calculate the total fitness score of the population.
- Calculate the selection probability for each individual by dividing its fitness score by the total fitness score.
- Calculate the cumulative probabilities by taking the cumulative sum of the selection probabilities.
- Select n_selection individuals by repeating the following steps:
a). Generate a random number r between 0 and 1.
b). Find the first individual whose cumulative probability is greater than r and select it.
- Return the selected individuals as a list.

In summary, this function uses the roulette wheel method to select n_selection individuals from the population based on their fitness scores, and returns them as a list.
- ### Crossover
```python
def single_point_crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring
```
The single_point_crossover function takes a list of parent individuals as input and returns a list of offspring individuals produced by applying single-point crossover.

The function loops through the parents list, selecting two parents at a time to apply the crossover operator. For each pair of parents, the function randomly selects a crossover point between the first and last gene of the parent's chromosome. The first offspring is created by combining the first part of the first parent's chromosome (from the beginning up to the crossover point) with the second part of the second parent's chromosome (from the crossover point to the end). The second offspring is created by combining the first part of the second parent's chromosome (from the beginning up to the crossover point) with the second part of the first parent's chromosome (from the crossover point to the end).

The resulting offspring are added to a list and returned at the end of the function.

- ### Mutation
```python
def mutate(offspring: list, mutation_rate: float = 0.01) -> list:
    for i in range(len(offspring)):
        if random.uniform(0, 1) < mutation_rate:
            j = random.randint(0, len(offspring[i]) - 1)
            offspring[i][j] = not offspring[i][j]

    return offspring
```
This function implements mutation in the genetic algorithm. It takes in a list of individuals (offspring) and a mutation rate as input parameters. The mutation rate determines the probability of a bit in an individual's chromosome to be flipped.

The function loops through all the individuals in the offspring list and checks if the mutation rate is less than a randomly generated value between 0 and 1. If it is, a random bit in the individual's chromosome is selected, and its value is flipped (i.e., if it is 0, it becomes 1, and if it is 1, it becomes 0).

Finally, the function returns the mutated offspring list.
