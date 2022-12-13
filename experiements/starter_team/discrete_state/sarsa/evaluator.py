import sys
sys.path.append('..')

from pol import GreedyPolicy, EpsilonPolicy
from env import RLPlayer, MaxDamagePlayer, action_to_showdown, showdown_to_switch
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
from pol import load_q

BATTLES = 100

with open('../team.txt', 'r') as f:
    team = f.read()

def evaluate(q, player, n_battle):
    pol = GreedyPolicy(q)

    battles = 0
    s, _, _, _, = player.step(0)
    print(f'Running battle: {battles}')
    while battles < n_battle:
        available_moves = [0, 1, 2, 3]
        available_switches_show = player.current_battle.available_switches
        available_switches_env = showdown_to_switch(available_switches_show)
        available_actions_env = available_moves + available_switches_env
        a = pol.act(s, available_actions_env)
        a_show = action_to_showdown(available_switches_show, a)
        sp, _, battle_over, _ = player.step(a_show)
        s = sp
        if battle_over:
            battles += 1
            player.reset()
            print(f'Running battle: {battles}')
    
    n_battles = player.n_finished_battles
    n_wins = player.n_won_battles

    return {
        'n_battles': n_battles,
        'n_wins': n_wins
    }

pc = PlayerConfiguration('OPPONENT0', '')
OPPONENT = RandomPlayer(
    battle_format="gen8ou",
    team=team
)

pc = PlayerConfiguration('PLAYER0', '')
PLAYER = RLPlayer(
    opponent=OPPONENT,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

q = load_q('results/final_modals/random_q.json')

test_results = evaluate(q, PLAYER, BATTLES)
print(f'Random: {test_results["n_wins"]}/{test_results["n_battles"]}')

pc = PlayerConfiguration('OPPONENT1', '')
OPPONENT = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER1', '')
PLAYER = RLPlayer(
    opponent=OPPONENT,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

q = load_q('results/final_modals/max_damage_q.json')

test_results = evaluate(q, PLAYER, BATTLES)
print(f'Max Damage: {test_results["n_wins"]}/{test_results["n_battles"]}')

pc = PlayerConfiguration('OPPONENT2', '')
OPPONENT = SimpleHeuristicsPlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER2', '')
PLAYER = RLPlayer(
    opponent=OPPONENT,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

q = load_q('results/final_modals/heuristics_q.json')

test_results = evaluate(q, PLAYER, BATTLES)
print(f'Heuristics: {test_results["n_wins"]}/{test_results["n_battles"]}')

