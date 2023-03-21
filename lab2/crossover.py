import random
from typing import List, Callable
from enum import Enum


class CrossOver(Enum):
    SPC = "SINGLE_PONT_CROSSOVER"
    MPC = "MIDDLE_POINT_CROSSOVER"

    def __str__(self):
        return self.value


def get_crossover(crossover: CrossOver) -> Callable:
    crossovers = {
        CrossOver.SPC: single_point_crossover,
        CrossOver.MPC: middle_point_crossover,
    }
    return crossovers[crossover]


def single_point_crossover(parents: List[List[bool]]) -> List[List[bool]]:
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring


def middle_point_crossover(parents: List[List[bool]]) -> List[List[bool]]:
    offspring = []
    crossover_point = int(len(parents) // 2)
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring
