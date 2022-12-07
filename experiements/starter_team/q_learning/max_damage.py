import sys
sys.path.append('..')

from poke_env.player import Player
from poke_env.player_configuration import PlayerConfiguration
from env import RLPlayerCustom
from algo import QLearning
from matplotlib import pyplot as plt
import json

EXPERIEMENT_NAME = 'max_damage_ql'
EXPERIEMENT_PATH = f'results/{EXPERIEMENT_NAME}'
TRAIN_STEPS = 500_000
EVAL_STEPS = 100

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

with open('../team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration(f'{EXPERIEMENT_NAME}_op', '')
player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration(EXPERIEMENT_NAME, '')
rl_player = RLPlayerCustom(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

ql = QLearning(gamma=1, e_start=0.1, a_start=0.1, e_dec=0, a_dec=0)
train_results = ql.train(rl_player, TRAIN_STEPS)

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

test_results = ql.eval(rl_player, EVAL_STEPS)
print(f'Eval: {test_results["n_wins"]}/{test_results["n_battles"]}')
with open(f'{EXPERIEMENT_PATH}/eval.json', 'w') as f:
    json.dump(test_results, f)

ql.save_q(f'{EXPERIEMENT_PATH}/q.pickle')
rl_player.close()
