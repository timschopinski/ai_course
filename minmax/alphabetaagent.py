from lab3.exceptions import GameplayException
from lab3.connect4 import Connect4
from lab3.randomagent import RandomAgent
from agent import Agent


class AlphaBetaAgent(Agent):
    def __init__(self, token, depth=5):
        super().__init__(token)
        self.depth = depth

    def decide(self, connect4):
        _, column = self.alpha_beta(
            connect4, self.depth, float("-inf"), float("inf"), True
        )
        return column

    def alpha_beta(self, connect4, depth, alpha, beta, maximize):
        if depth == 0 or connect4.game_over:
            return self._evaluate_board(connect4), None

        if maximize:
            best_value = float("-inf")
            best_column = None
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value, _ = self.alpha_beta(connect4, depth - 1, alpha, beta, False)
                connect4.undo(n_column)
                if value > best_value:
                    best_value = value
                    best_column = n_column
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_value, best_column

        else:
            best_value = float("-inf")
            best_column = None
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value, _ = self.alpha_beta(connect4, depth - 1, alpha, beta, True)
                connect4.undo(n_column)
                if value > best_value:
                    best_value = value
                    best_column = n_column
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
            return best_value, best_column

    def _evaluate_board(self, connect4):
        score = 0
        for four in connect4.iter_fours():
            if four.count(self.opponent_token) == 4:
                return -1000000
            elif four.count(self.my_token) == 4:
                return 1000000
            elif four.count(self.my_token) == 3 and four.count("_") == 1:
                score += 100
            elif four.count(self.opponent_token) == 3 and four.count("_") == 1:
                score -= 100
            elif four.count(self.my_token) == 2 and four.count("_") == 2:
                score += 10
            elif four.count(self.opponent_token) == 2 and four.count("_") == 2:
                score -= 10
        return score

    def simulate(self, game, n_column, token):
        next_game = Connect4(width=game.width, height=game.height)
        next_game.board = [row[:] for row in game.board]
        next_game.who_moves = game.who_moves
        next_game.game_over = game.game_over
        next_game.wins = game.wins
        try:
            next_game.drop_token(n_column)
        except GameplayException:
            pass
        return next_game
