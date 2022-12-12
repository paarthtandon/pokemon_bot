from algo import DQNN
from env import RLPlayer, MaxDamagePlayer
from poke_env.player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

MODEL_CHECKPOINT = 'results/max_damage_dqn/checkpoints/model_final.pt'
EPISODES = 100

with open('team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration('OPPONENT', '')
player = RandomPlayer(
    battle_format="gen8ou",
    team=team
)
player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team
)

pc = PlayerConfiguration('PLAYER', '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team
)

model = DQNN()
model.load_checkpoint(MODEL_CHECKPOINT)
results = model.evaluate(rl_player, EPISODES)

print(results)
