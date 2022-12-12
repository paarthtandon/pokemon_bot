from tqdm import tqdm
from pol import EpsilonPolicy, GreedyPolicy
import ujson
from env import showdown_to_switch, action_to_showdown

class QLearning:

    def __init__(self, gamma, e_start, a_start, e_dec, a_dec, min_e=0, min_a=0, q=None):
        self.gamma = gamma
        self.e_start = e_start
        self.a_start = a_start
        self.e = e_start
        self.a = a_start
        self.min_e = min_e
        self.min_a = min_a
        self.e_dec = e_dec
        self.a_dec = a_dec
        self.it = 0

        if q:
            self.q = q
        else:
            self.q = {}
            for mp in range(3):
                for op in range(3):
                    for mh in range(4):
                        for oh in range(4):
                            self.q[(mp, op, mh, oh)] = {}
                            for a in range(7):
                                self.q[(mp, op, mh, oh)][a] = 0
        
        self.pol = EpsilonPolicy(self.q, self.e)
    
    def q_step(self, s, a, r, sp):
        s = tuple(s)
        sp = tuple(sp)
        max_q = max(list(self.q[sp].values()))
        self.q[s][a] += self.a * (r + (self.gamma * max_q) - self.q[s][a])

        self.e = max(self.min_e, self.e - self.e_dec)
        self.a = max(self.min_a, self.a - self.a_dec)
        self.pol.pol = self.q
        self.pol.e = self.e
    
    def train(self, player, n_steps):
        rewards = []
        win_rate = []
        steps = []
        cur_reward = 0

        s, _, battle_over, _, = player.step(0)
        for i in tqdm(range(n_steps)):
            if player.current_battle.available_moves:
                available_moves = [0, 1, 2, 3]
            else:
                available_moves = []
            available_switches_show = player.current_battle.available_switches
            available_switches_env = showdown_to_switch(available_switches_show)
            available_actions_env = available_moves + available_switches_env
            a = self.pol.act(s, available_actions_env)
            a_show = action_to_showdown(available_switches_show, a)
            if battle_over:
                player.reset()
            sp, r, battle_over, _ = player.step(a_show)
            self.q_step(s, a, r, sp)
            s = sp
            cur_reward += r
            
            try:
                n_battles = player.n_finished_battles
                n_wins = player.n_won_battles
                if n_battles != 0:
                    win_r = n_wins / n_battles
                else:
                    win_r = 0
                if n_battles != 0:
                    win_rate.append(win_r)
                else:
                    win_rate.append(0)
                steps.append(i)
                rewards.append(cur_reward)
                if self.it % 1000 == 0:
                    print(f'Current win rate: {win_r} ({n_wins}/{n_battles}) e: {self.e}')
                self.it += 1
            except Exception as e:
                print('failed step', e)
                continue
        
        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'steps': steps,
            'rewards': rewards,
            'win_rate': win_rate,
            'n_battles': n_battles,
            'n_wins': n_wins
        }
    
    def save_q(self, fname):
        with open(fname, 'w') as f:
            ujson.dump(self.q, f)


class SARSA:

    def __init__(self, gamma, e_start, a_start, e_dec, a_dec, min_e=0, min_a=0, q=None):
        self.gamma = gamma
        self.e_start = e_start
        self.a_start = a_start
        self.e = e_start
        self.a = a_start
        self.min_e = min_e
        self.min_a = min_a
        self.e_dec = e_dec
        self.a_dec = a_dec
        self.it = 0

        if q:
            self.q = q
        else:
            self.q = {}
            for mp in range(3):
                for op in range(3):
                    for mh in range(4):
                        for oh in range(4):
                            self.q[(mp, op, mh, oh)] = {}
                            for a in range(6):
                                self.q[(mp, op, mh, oh)][a] = 0
        
        self.pol = EpsilonPolicy(self.q, self.e)
    
    def q_step(self, s, a, r, sp):
        s = tuple(s)
        sp = tuple(sp)
        ap = self.pol.act(sp)
        self.q[s][a] += self.a * (r + (self.gamma * self.q[sp][ap]) - self.q[s][a])

        self.e = max(self.min_e, self.e - self.e_dec)
        self.a = max(self.min_a, self.a - self.a_dec)
        self.pol.pol = self.q
        self.pol.e = self.e
        self.it += 1
    
    def train(self, player, n_steps):
        rewards = []
        steps = []
        cur_reward = 0

        s, _, _, _, = player.step(0)
        for i in tqdm(range(n_steps)):
            a = self.pol.act(s)
            sp, r, battle_over, _ = player.step(a)
            self.q_step(s, a, r, sp)
            s = sp
            cur_reward += r
            rewards.append(cur_reward)
            steps.append(i)
            if battle_over:
                player.reset()
        
        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'steps': steps,
            'rewards': rewards,
            'n_battles': n_battles,
            'n_wins': n_wins
        }
    
    def eval(self, player, n_battle):
        player.reset_env()
        pol = GreedyPolicy(self.pol.pol)

        battles = 0
        s, _, _, _, = player.step(0)
        while battles < n_battle:
            a = pol.act(s)
            s, _, over, _ = player.step(a)
            if over:
                battles += 1
                player.reset()
        
        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles

        return {
            'n_battles': n_battles,
            'n_wins': n_wins
        }
    
    def save_q(self, fname):
        f = open(fname, 'ab')
        pickle.dump(self.q, f)
        f.close()
