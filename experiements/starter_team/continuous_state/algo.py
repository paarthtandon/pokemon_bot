import torch
from torch import nn
from collections import deque
import random
from env import action_to_showdown, showdown_to_switch
import copy
from tqdm import tqdm
from memory_profiler import profile

class Sample:
    def __init__(
        self,
        state,
        action,
        reward,
        state_p,
        terminal
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_p = state_p
        self.terminal = terminal

class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, action_dim),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class DQNN:
    def __init__(
        self,
        memory_size=10000,
        minibatch_size=32,
        state_dim=8,
        action_dim=7,
        reset_Q_steps=1000,
        gamma=1,
        epsilon_start=0.1,
        epsilon_min = 0.1,
        epsolon_dec = 0,
        lr=0.01,
        checkpoint_path=None
    ):
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.D = deque([], memory_size)
        self.Q = Q(state_dim, action_dim)
        self.Q_hat = Q(state_dim, action_dim)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.reset_Q_steps = reset_Q_steps
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsolon_dec
        self.optimizer = torch.optim.SGD(self.Q.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.checkpoint_path = checkpoint_path
        self.losses = []
        self.loss_means = []

    def checkpoint(self, eps):
        torch.save(self.Q, f'{self.checkpoint_path}/model_{eps}.pt')
    
    def load_checkpoint(self, fname):
        self.Q = torch.load(fname)

    def store_sample(self, sample):
        self.D.append(sample)

    def sample_minibatch(self):
        return random.sample(list(self.D), min(len(list(self.D)), self.minibatch_size))

    def td_target(self, sample):
        if sample.terminal:
            return sample.reward
        state_p = torch.tensor(sample.state_p)
        q_values = self.Q_hat(state_p)
        q_max = torch.max(q_values)
        return sample.reward + (self.gamma * q_max)

    def gradient_step(self):
        batch = self.sample_minibatch()
        targets = torch.tensor([self.td_target(s) for s in batch])

        states = torch.stack([s.state for s in batch])

        preds_q = self.Q(states)
        preds = []
        for i in range(len(batch)):
            preds.append(preds_q[i, batch[i].action])
        preds = torch.tensor(preds, requires_grad=True)
        loss = self.loss(preds, targets)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)
    
    def gradient_step_single(self, sample):
        target = torch.tensor(self.td_target(sample))
        state = sample.state
        pred = self.Q(state)[sample.action]
        loss = torch.square(pred - target)
        # print(target, pred)
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

    def reset_Q_hat(self):
        del self.Q_hat
        self.Q_hat = Q(self.state_dim, self.action_dim)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.Q_hat.eval()
    
    def choose_action(self, player, state, epsilon=None):
        possible_moves = range(len(player.current_battle.available_moves))
        possible_switches = player.current_battle.available_switches
        possible_switch_actions = showdown_to_switch(possible_switches)
        possible_actions = list(possible_moves) + possible_switch_actions

        if epsilon:
            if random.random() < epsilon:
                action = random.choice(possible_actions)
                return action, action_to_showdown(possible_switches, action)
        else:
            if random.random() < self.epsilon:
                action = random.choice(possible_actions)
                return action, action_to_showdown(possible_switches, action)
        
        with torch.no_grad():
            q_vals = self.Q(state)
        
        mx = float('-inf')
        mx_a = None
        for a in possible_actions:
            if q_vals[a].item() > mx:
                mx = q_vals[a].item()
                mx_a = a

        return mx_a, action_to_showdown(possible_switches, mx_a)
    
    def train(self, player, episodes):
        self.Q.train()
        rewards = []
        win_rate = []
        actions = {}
        eps = []
        cur_reward = 0
        steps = 0
        
        Q_hat_counter = 0
        for ep in tqdm(range(episodes)):
            state, _, battle_over, _, = player.step(0)
            while not battle_over:
                state = torch.tensor(state)
                action, action_show = self.choose_action(player, state)

                if action not in actions:
                    actions[action] = 0
                actions[action] += 1

                state_p, reward, battle_over, _ = player.step(action_show)
                sample = Sample(
                    state,
                    action,
                    reward,
                    state_p,
                    battle_over
                )
                self.store_sample(sample)
                steps += 1
                
                self.gradient_step()
                state = state_p

                Q_hat_counter += 1
                if Q_hat_counter % self.reset_Q_steps == 0:
                    self.reset_Q_hat()
                    Q_hat_counter = 0
                
                cur_reward += reward
                rewards.append(cur_reward)
                if player.n_finished_battles != 0:
                    win_rate.append(
                        player.n_won_battles / player.n_finished_battles
                    )
                else:
                    win_rate.append(0)
                eps.append(ep)
                self.loss_means.append(sum(self.losses) / len(self.losses))

            if ep % 10 == 0 and len(win_rate) > 0:
                print(f'Current win rate: {win_rate[-1]} ({player.n_won_battles}/{player.n_finished_battles}) D: {len(self.D)}')
                action_p = sorted(
                    list(actions.items()),
                    key=lambda x: x[0]
                )
                action_p = [(a[0], round(a[1] / sum(actions.values()), 4)) for a in action_p]
                print(f'Steps: {steps}, e: {self.epsilon}\nactions: {action_p}')
            if ep % 1000 == 0:
                print('Saved checkpoint')
                self.checkpoint(ep)
            player.reset()
        self.checkpoint('final')

        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'episodes': eps,
            'rewards': rewards,
            'losses': self.loss_means,
            'win_rate': win_rate,
            'n_battles': n_battles,
            'n_wins': n_wins
        }

    def evaluate(self, player, episodes):
        self.Q.eval()

        for ep in range(episodes):
            print(f'Running battle: {ep}')
            state, _, battle_over, _, = player.step(0)
            while not battle_over:
                state = torch.tensor(state)
                action, action_show = self.choose_action(player, state)
                state, _, battle_over, _ = player.step(action_show)
            player.reset()

        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'n_battles': n_battles,
            'n_wins': n_wins
        }