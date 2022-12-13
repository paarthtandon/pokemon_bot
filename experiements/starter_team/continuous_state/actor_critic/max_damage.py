import sys
sys.path.append('..')

from env import RLPlayer, MaxDamagePlayer
from poke_env.player_configuration import PlayerConfiguration
from algo import AC
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'max_damage_ac'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_EPISODES = 5000
EVAL_EPISODES = 10

with open('team.txt', 'r') as teamf:
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

action_space = rl_player.action_space.n
print(action_space)

learner = AC(
    state_dim=8,
    action_dim=7,
    gamma=0.5,
    actor_lr=1e-3,
    critic_lr=1e-3,
    checkpoint_path=f'{EXPERIEMENT_PATH}/checkpoints'
)
train_results = learner.train(rl_player, TRAIN_EPISODES)

plt.title(f'AC vs. {EXPERIEMENT_NAME}')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(train_results['episodes'], train_results['rewards'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_rewards.png')

plt.cla()
plt.title(f'AC vs. {EXPERIEMENT_NAME}')
plt.xlabel('episodes')
plt.ylabel('win rate')
plt.plot(train_results['episodes'], train_results['win_rate'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_win_r.png')

plt.cla()
plt.title(f'Critic Loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.plot(range(len(train_results['critic_losses'])), train_results['critic_losses'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_closs.png')

plt.cla()
plt.title(f'Actor Loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.plot(range(len(train_results['actor_losses'])), train_results['actor_losses'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_aloss.png')

with open(f'{EXPERIEMENT_PATH}/train.json', 'w') as f:
    json.dump(train_results, f)

rl_player.close()
