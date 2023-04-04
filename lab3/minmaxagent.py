from agent import Agent


class MinMaxAgent(Agent):
    def __init__(self, token, depth=5):
        super().__init__(token)
        self.depth = depth

    def decide(self, connect4):
        _, n_column = self._minimax(connect4, self.depth, True)
        return n_column

    def _minimax(self, connect4, depth, maximize):
        if depth == 0 or connect4.game_over:
            return self._evaluate_board(connect4), None

        if maximize:
            best_value = float("-inf")
            best_column = None
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value, _ = self._minimax(connect4, depth - 1, False)
                connect4.undo(n_column)
                if value > best_value:
                    best_value = value
                    best_column = n_column
            return best_value, best_column

        else:
            best_value = float("inf")
            best_column = None
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value, _ = self._minimax(connect4, depth - 1, True)
                connect4.undo(n_column)
                if value < best_value:
                    best_value = value
                    best_column = n_column
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
