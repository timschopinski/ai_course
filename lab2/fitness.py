from itertools import compress


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items["Weight"], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items["Value"], individual))
