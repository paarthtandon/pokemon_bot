from pol import GreedyPolicy
from env import RLPlayer
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
import pickle


Q_PATH = 'sarsa/results/random_sarsa/q.pickle'
BATTLES = 100

with open(Q_PATH, 'rb') as f:
    q = pickle.load(f)
with open('team.txt', 'r') as f:
    team = f.read()

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

def evaluate(q, player, n_battle):
    pol = GreedyPolicy(q)

    battles = 0
    s, _, _, _, = player.step(0)
    print(f'Running battle: {battles}')
    while battles < n_battle:
        a = pol.act(s)
        s, _, over, _ = player.step(a)
        if over:
            battles += 1
            player.reset()
            print(f'Running battle: {battles}')
    
    n_battles = player.n_finished_battles
    n_wins = player.n_won_battles

    return {
        'n_battles': n_battles,
        'n_wins': n_wins
    }

OPPONENT = RandomPlayer(
    battle_format="gen4ou",
    team=team
)

# OPPONENT = MaxDamagePlayer(
#     battle_format="gen4ou",
#     team=team
# )

# OPPONENT = SimpleHeuristicsPlayer(
#     battle_format="gen8ou",
#     team=team
# )

pc = PlayerConfiguration('PLAYER', '')
PLAYER = RLPlayer(
    opponent=OPPONENT,
    battle_format="gen4ou",
    team=team,
    player_configuration=pc
)


test_results = evaluate(q, PLAYER, BATTLES)
print(f'Eval: {test_results["n_wins"]}/{test_results["n_battles"]}')
