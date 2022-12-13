import sys
sys.path.append('..')

from algo import DQNN
from env import RLPlayer, MaxDamagePlayer
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration

EPISODES = 100

with open('team.txt', 'r') as teamf:
    team = teamf.read()

pc = PlayerConfiguration('OPPONENT0', '')
player = RandomPlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER0', '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

model = DQNN()
model.load_checkpoint('results/final_modals/random_final.pt')
results = model.evaluate(rl_player, EPISODES)

print('Random:')
print(results)

pc = PlayerConfiguration('OPPONENT1', '')
player = MaxDamagePlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER1', '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

model = DQNN()
model.load_checkpoint('results/final_modals/max_damage_final.pt')
results = model.evaluate(rl_player, EPISODES)

print('Max:')
print(results)

pc = PlayerConfiguration('OPPONENT2', '')
player = SimpleHeuristicsPlayer(
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

pc = PlayerConfiguration('PLAYER2', '')
rl_player = RLPlayer(
    opponent=player,
    battle_format="gen8ou",
    team=team,
    player_configuration=pc
)

model = DQNN()
model.load_checkpoint('results/final_modals/heuristics_final.pt')
results = model.evaluate(rl_player, EPISODES)

print('Heuristics:')
print(results)
