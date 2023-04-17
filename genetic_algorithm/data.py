import pandas as pd
from pandas import DataFrame


def get_small() -> tuple[DataFrame, int]:
    knapsack = pd.read_csv("knapsack-small.csv")
    return knapsack, 10


def get_big() -> tuple[DataFrame, int]:
    knapsack = pd.read_csv("knapsack-big.csv")
    return knapsack, 6404180
