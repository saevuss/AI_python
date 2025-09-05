from easyAI import TwoPlayerGame, Human_Player, AI_Player
from easyAI.AI import Negamax, TranspositionTable

class LastCoin_game(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.nplayer = 1
        self.num_coins = 15
        self.max_coins = 4

    def possible_moves(self):
        return[str(a) for a in range(1, self.max_coins + 1)]

    def make_move(self, move):
        self.num_coins -= int(move)

    def win_game(self):
        return self.num_coins <=0

    def is_over(self):
        return self.win_game()

    def score(self):
        return 100 if self.win_game() else 0

    def show(self):
        print(self.num_coins, ' coins left in the pile')

if __name__ == '__main__':
    tt = TranspositionTable()
    LastCoin_game.ttentry = lambda self: self.num_coins
    #solving the game
    solver = Negamax(15, tt=tt)
    ai_player = AI_Player(solver)
    dummy_game = LastCoin_game([AI_Player(solver), AI_Player(solver)])
    move = ai_player._compute_ai_move(dummy_game)
    print("Mossa iniziale migliore: ", move)
    #deciding who will start the game
    game = LastCoin_game([AI_Player(tt), Human_Player()])
    game.play()
