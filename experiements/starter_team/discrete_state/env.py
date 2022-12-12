import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer, Player

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
            current_battle, hp_value=1.0, fainted_value=10.0, victory_value=100.0
        )

    def embed_battle(self, battle):
        def hp_bin(hp):
            return int(3 * hp)

        pokemon = {
            'charizard': 0,
            'venusaur': 1,
            'blastoise': 2
        }

        my_mon = pokemon[battle.active_pokemon.species]
        opponent_mon = pokemon[battle.opponent_active_pokemon.species]

        my_hp = hp_bin(battle.active_pokemon.current_hp_fraction)
        opponent_hp = hp_bin(battle.opponent_active_pokemon.current_hp_fraction)

        return [my_mon, opponent_mon, my_hp, opponent_hp]
    
    def describe_embedding(self) -> Space:
        low = [0, 0, 0, 0]
        high = [0, 0, 3, 3]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

class RLPlayerCustom(Gen8EnvSinglePlayer):

    def calc_reward(self, last_battle, current_battle) -> float:
        reward = 0

        current_fainted = 0
        for mon in current_battle.opponent_team.values():
            if mon.fainted:
                current_fainted += 1
        
        reward = reward + (current_fainted * 20)

        current_fainted = 0
        for mon in current_battle.opponent_team.values():
            if mon.fainted:
                current_fainted += 1
        
        reward = reward + (current_fainted * -10)
        
        if current_battle.won:
            reward += 1000
        elif current_battle.lost:
            reward -= 1000
        
        return reward

    def embed_battle(self, battle):
        def hp_bin(hp):
            return int(3 * hp)

        pokemon = {
            'charizard': 0,
            'venusaur': 1,
            'blastoise': 2
        }

        my_mon = pokemon[battle.active_pokemon.species]
        opponent_mon = pokemon[battle.opponent_active_pokemon.species]

        my_hp = hp_bin(battle.active_pokemon.current_hp_fraction)
        opponent_hp = hp_bin(battle.opponent_active_pokemon.current_hp_fraction)

        return [my_mon, opponent_mon, my_hp, opponent_hp]
    
    def describe_embedding(self) -> Space:
        low = [0, 0, 0, 0]
        high = [0, 0, 3, 3]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
