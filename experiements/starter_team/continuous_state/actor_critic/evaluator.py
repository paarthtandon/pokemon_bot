import sys
sys.path.append('..')

from algo import AC
from env import RLPlayer, MaxDamagePlayer
from poke_env.player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

MODEL_CHECKPOINT_ACT = 'results/max_damage_ac/checkpoints/model_a3000.pt'
MODEL_CHECKPOINT_CRT = 'results/max_damage_ac/checkpoints/model_c3000.pt'
EPISODES = 100

with open('team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration('OPPONENT', '')

# player = RandomPlayer(
#     battle_format="gen8ou",
#     team=team,
#     player_configuration=pc
# )

player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER', '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

model = AC()
model.load_checkpoint(MODEL_CHECKPOINT_ACT, MODEL_CHECKPOINT_CRT)
results = model.evaluate(rl_player, EPISODES)

print(results)
