import random
from typing import List


def mutate(offspring: List[List[bool]], mutation_rate: float = 0.01) -> List[List[bool]]:
    for i in range(len(offspring)):
        if random.uniform(0, 1) < mutation_rate:
            j = random.randint(0, len(offspring[i]) - 1)
            offspring[i][j] = not offspring[i][j]

    return offspring
