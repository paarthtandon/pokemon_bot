import sys
sys.path.append('..')

from poke_env.player import Player
from poke_env.player_configuration import PlayerConfiguration
from env import RLPlayerCustom, RLPlayer, MaxDamagePlayer
from algo import QLearning
from pol import load_q
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'max_damage_ql'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_STEPS = 25_000

with open('../team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration(f'{EXPERIEMENT_NAME}_op', '')
player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration(EXPERIEMENT_NAME, '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

ql = QLearning(gamma=0.5, e_start=0.01, a_start=0.1, e_dec=0, a_dec=0)
train_results = ql.train(rl_player, TRAIN_STEPS)
ql.save_q(f'{EXPERIEMENT_PATH}/q.json')

plt.title(f'QL vs. {EXPERIEMENT_NAME}')
plt.xlabel('steps')
plt.ylabel('reward')
plt.plot(train_results['steps'], train_results['rewards'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_rewards.png')

plt.cla()
plt.title(f'QL vs. {EXPERIEMENT_NAME}')
plt.xlabel('steps')
plt.ylabel('win rate')
plt.plot(train_results['steps'], train_results['win_rate'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_win_r.png')

with open(f'{EXPERIEMENT_PATH}/train.json', 'w') as f:
    json.dump({
        'steps': train_results['steps'],
        'rewards': train_results['rewards'],
        'win_rate': train_results['win_rate']
    }, f)

rl_player.close()
