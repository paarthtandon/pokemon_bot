from pol import GreedyPolicy
from env import RLPlayer, MaxDamagePlayer
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
import pickle


Q_PATH = 'q_learning/results/max_damage_ql/q.pickle'
BATTLES = 100

with open(Q_PATH, 'rb') as f:
    q = pickle.load(f)
with open('team.txt', 'r') as f:
    team = f.read()

print(q)

def evaluate(q, player, n_battle):
    pol = GreedyPolicy(q)

    battles = 0
    s, _, _, _, = player.step(0)
    print(f'Running battle: {battles}')
    while battles < n_battle:
        if player.current_battle.available_moves:
            a = pol.act(s)
        else:
            a = pol.act(s, only_switch=True)
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

# OPPONENT = RandomPlayer(
#     battle_format="gen8ou",
#     team=team
# )

OPPONENT = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team
)

# OPPONENT = SimpleHeuristicsPlayer(
#     battle_format="gen8ou",
#     team=team
# )

pc = PlayerConfiguration('PLAYER', '')
PLAYER = RLPlayer(
    opponent=OPPONENT,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)


test_results = evaluate(q, PLAYER, BATTLES)
print(f'Eval: {test_results["n_wins"]}/{test_results["n_battles"]}')
