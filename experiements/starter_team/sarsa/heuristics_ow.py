import sys
sys.path.append('..')

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration
from env import RLPlayerOnlyWins
from algo import SARSA
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'heuristics_ow_sarsa'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_STEPS = 500_000
EVAL_STEPS = 100

with open('../team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration(f'heur_ow_srs_op', '')
player = SimpleHeuristicsPlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration(f'heur_ow_srs', '')
rl_player = RLPlayerOnlyWins(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

ql = SARSA(gamma=1, e_start=1, a_start=1, e_dec=1/(TRAIN_STEPS/2), a_dec=1/(TRAIN_STEPS/2), min_e=0.01, min_a=0.01)
train_results = ql.train(rl_player, TRAIN_STEPS)

plt.title(f'QL vs. {EXPERIEMENT_NAME}')
plt.xlabel('steps')
plt.ylabel('reward')
plt.plot(train_results['steps'], train_results['rewards'])
plt.savefig(f'{EXPERIEMENT_PATH}/training.png')

with open(f'{EXPERIEMENT_PATH}/train.json', 'w') as f:
    json.dump({
        'steps': train_results['steps'],
        'rewards': train_results['rewards']
    }, f)

test_results = ql.eval(rl_player, EVAL_STEPS)
print(f'Eval: {test_results["n_wins"]}/{test_results["n_battles"]}')
with open(f'{EXPERIEMENT_PATH}/eval.json', 'w') as f:
    json.dump(test_results, f)

ql.save_q(f'{EXPERIEMENT_PATH}/q.pickle')
rl_player.close()
