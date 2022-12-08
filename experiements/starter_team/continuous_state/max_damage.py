from env import RLPlayer, MaxDamagePlayer
from poke_env.player_configuration import PlayerConfiguration
from algo import DQNN
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'max_damage_dqn'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_EPISODES = 1000
EVAL_EPISODES = 10

with open('team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration(f'{EXPERIEMENT_NAME}_op', '')
player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team
)

pc = PlayerConfiguration(EXPERIEMENT_NAME, '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team
)

learner = DQNN()
train_results = learner.train(rl_player, TRAIN_EPISODES)

plt.title(f'DQN vs. {EXPERIEMENT_NAME}')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(train_results['episodes'], train_results['rewards'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_rewards.png')

plt.cla()
plt.title(f'DQN vs. {EXPERIEMENT_NAME}')
plt.xlabel('episodes')
plt.ylabel('win rate')
plt.plot(train_results['episodes'], train_results['win_rate'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_win_r.png')

with open(f'{EXPERIEMENT_PATH}/train.json', 'w') as f:
    json.dump({
        'episodes': train_results['episodes'],
        'rewards': train_results['rewards'],
        'win_rate': train_results['win_rate']
    }, f)

rl_player.close()
