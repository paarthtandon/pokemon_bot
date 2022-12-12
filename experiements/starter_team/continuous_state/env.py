from poke_env.player import Gen8EnvSinglePlayer, Player
import numpy as np
from gym.spaces import Space, Box
import torch

def showdown_to_switch(switches):
    pokemon = {
        'charizard': 4,
        'venusaur': 5,
        'blastoise': 6
    }
    available = [pokemon[p.species] for p in switches]
    return available

def action_to_showdown(switches, action):
    if action < 4:
        return action
    pokemon = {
        4: 'charizard',
        5: 'venusaur',
        6: 'blastoise'
    }
    for i in range(len(switches)):
        if pokemon[action] == switches[i].species:
            return 16 + i


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

class RLPlayer(Gen8EnvSinglePlayer):

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, hp_value=1.0, fainted_value=0.0, victory_value=0.0
        )

    def embed_battle(self, battle):
        embedding = []

        pokemon = {
            'charizard': 0,
            'venusaur': 1,
            'blastoise': 2
        }

        my_active = battle.active_pokemon
        embedding.append(pokemon[my_active.species] / 2)

        my_team = list(battle.team.values())
        my_team_hp = [1, 1, 1]
        for p in my_team:
            i = pokemon[p.species]
            my_team_hp[i] = p.current_hp_fraction
        embedding += my_team_hp

        op_active = battle.opponent_active_pokemon
        embedding.append(pokemon[op_active.species] / 2)

        op_team = list(battle.opponent_team.values())
        op_team_hp = [1, 1, 1]
        for p in op_team:
            i = pokemon[p.species]
            op_team_hp[i] = p.current_hp_fraction
        embedding += op_team_hp

        return np.array(embedding, dtype=np.float32)

    # def embed_battle(self, battle):
    #     def hp_bin(hp):
    #         return int(3 * hp)

    #     pokemon = {
    #         'charizard': 0.0,
    #         'venusaur': 0.5,
    #         'blastoise': 1.0
    #     }

    #     my_mon = pokemon[battle.active_pokemon.species]
    #     opponent_mon = pokemon[battle.opponent_active_pokemon.species]

    #     my_hp = battle.active_pokemon.current_hp_fraction
    #     opponent_hp = battle.opponent_active_pokemon.current_hp_fraction

    #     return np.array([my_mon, opponent_mon, my_hp, opponent_hp], dtype=np.float32)
    
    def describe_embedding(self) -> Space:
        low = [0, 0, 0, 0, 0, 0, 0, 0]
        high = [2, 1, 1, 1, 2, 1, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
