from pol import GreedyPolicy
from env import RLPlayer, MaxDamagePlayer
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
from pol import load_q


Q_PATH = 'q_learning/results/random_ql/q.json'
BATTLES = 100

with open('team.txt', 'r') as f:
    team = f.read()
q = load_q(Q_PATH)
print(q)

def evaluate(q, player, n_battle):
    pol = GreedyPolicy(q)

    battles = 0
    s, _, _, _, = player.step(0)
    print(f'Running battle: {battles}')
    while battles < n_battle:
        if player.current_battle.available_moves:
            a = pol.act(s, player.current_battle.available_switches)
        else:
            a = pol.act(s, player.current_battle.available_switches, only_switch=True)
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
    battle_format="gen8ou",
    team=team
)

# OPPONENT = MaxDamagePlayer(
#     battle_format="gen8ou",
#     team=team
# )

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
