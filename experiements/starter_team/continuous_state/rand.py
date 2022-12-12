from env import RLPlayer
from poke_env.player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from algo import DQNN
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'random_dqn'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_EPISODES = 1000
EVAL_EPISODES = 10

with open('team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration(f'{EXPERIEMENT_NAME}_op', '')
player = RandomPlayer(
    battle_format="gen8ou",
    team=team
)

pc = PlayerConfiguration(EXPERIEMENT_NAME, '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team
)

"""
def calc_reward(self, last_battle, current_battle) -> float:
    return self.reward_computing_helper(
        current_battle, hp_value=1.0, fainted_value=10.0, victory_value=100.0
    )
"""

action_space = rl_player.action_space.n
print(action_space)

learner = DQNN(
    memory_size=int(1e5),
    minibatch_size=32,
    state_dim=8,
    action_dim=7,
    reset_Q_steps=400,
    gamma=0.5,
    epsilon_start=1,
    epsilon_min = 0.01,
    epsolon_dec = 0.9995,
    lr=1e-4,
    checkpoint_path=f'{EXPERIEMENT_PATH}/checkpoints'
)
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

plt.cla()
plt.title(f'DQN Loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.plot(range(len(train_results['losses'])), train_results['losses'])
plt.savefig(f'{EXPERIEMENT_PATH}/train_loss.png')

with open(f'{EXPERIEMENT_PATH}/train.json', 'w') as f:
    json.dump({
        'episodes': train_results['episodes'],
        'rewards': train_results['rewards'],
        'win_rate': train_results['win_rate']
    }, f)

rl_player.close()
