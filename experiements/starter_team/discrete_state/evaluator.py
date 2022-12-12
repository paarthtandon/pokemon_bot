from pol import GreedyPolicy, EpsilonPolicy
from env import RLPlayer, MaxDamagePlayer, action_to_showdown, showdown_to_switch
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
from pol import load_q


Q_PATH = 'q_learning/results/max_damage_ql/q.json'
BATTLES = 100

with open('team.txt', 'r') as f:
    team = f.read()
q = load_q(Q_PATH)

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

pc = PlayerConfiguration('OPPONENT', '')
# OPPONENT = RandomPlayer(
#     battle_format="gen8ou",
#     team=team
# )

OPPONENT = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
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
