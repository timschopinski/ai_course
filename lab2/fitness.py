from itertools import compress
from pandas import DataFrame
from typing import List


def fitness(items: DataFrame, knapsack_max_capacity: int, individual: List[bool]) -> float:
    total_weight = sum(compress(items["Weight"], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items["Value"], individual))
