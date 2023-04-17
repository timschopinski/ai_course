from lab3.exceptions import GameplayException
from lab3.connect4 import Connect4

connect4 = Connect4()
while not connect4.game_over:
    connect4.draw()
    try:
        n_column = int(input(':'))
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

if __name__ == "__main__":
    connect4.draw()
