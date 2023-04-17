from lab3.exceptions import GameplayException
from lab3.connect4 import Connect4
from lab3.randomagent import RandomAgent
from lab3.minmaxagent import MinMaxAgent

connect4 = Connect4(width=7, height=6)
agent = MinMaxAgent('x')
while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent.my_token:
            n_column = agent.decide(connect4)
        else:
            n_column = int(input(':'))
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

if __name__ == "__main__":
    connect4.draw()
