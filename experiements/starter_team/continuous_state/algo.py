import torch
from torch import nn
from collections import deque
import random
from env import action_to_showdown, showdown_to_switch
import copy

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
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
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
        state_dim=9,
        action_dim=7,
        reset_Q_steps=1000,
        gamma=1,
        epsilon=0.1,
        lr=0.01,
        checkpoint_path=None
    ):
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.D = deque([])
        self.Q = Q(state_dim, action_dim)
        self.Q_hat = Q(state_dim, action_dim)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.reset_Q_steps = reset_Q_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = torch.optim.SGD(self.Q.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.losses = []

    def checkpoint(self, eps):
        torch.save(self.Q, f'{self.checkpoint_path}/model_{eps}.pt')
    
    def load_checkpoint(self, fname):
        self.Q = torch.load(fname)

    def store_sample(self, sample):
        if len(self.D) >= self.memory_size:
            self.D.pop()
        self.D.append(sample)

    def sample_minibatch(self):
        return random.sample(self.D, min(len(self.D), self.minibatch_size))

    def td_target(self, sample):
        if sample.terminal:
            return sample.reward
        self.Q_hat.to(self.device)
        state_p = torch.tensor(sample.state_p).to(self.device)
        q_values = self.Q_hat(state_p)
        q_max = torch.max(q_values)
        #print(sample.reward + (self.gamma * q_max))
        return sample.reward + (self.gamma * q_max)

    def gradient_step(self):
        self.Q.to(self.device)
        self.Q_hat.to(self.device)

        batch = self.sample_minibatch()
        targets = torch.tensor([[self.td_target(s)]*self.action_dim for s in batch]).to(self.device)
        preds = self.Q(torch.stack([s.state for s in batch])).to(self.device)
        loss = self.loss(preds, targets)
        self.losses.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_Q_hat(self):
        self.Q_hat = copy.deepcopy(self.Q)
    
    def choose_action(self, player, state, epsilon=None):
        possible_moves = range(len(player.current_battle.available_moves))
        possible_switches = player.current_battle.available_switches
        possible_switch_actions = showdown_to_switch(possible_switches)
        possible_actions = list(possible_moves) + possible_switch_actions

        if epsilon:
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
        else:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
        
        self.Q.to(self.device)
        state.to(self.device)
        q_vals = self.Q(state)
        mx = float('-inf')
        mx_a = None
        for a in possible_actions:
            if q_vals[a] > mx:
                mx = q_vals[a]
                mx_a = a
        return action_to_showdown(possible_switches, mx_a)
    
    def train(self, player, episodes):
        self.Q.train()
        rewards = []
        win_rate = []
        eps = []
        cur_reward = 0
        steps = 0
        
        Q_hat_counter = 0
        for ep in range(episodes):
            state, _, battle_over, _, = player.step(0)
            while not battle_over:
                state = torch.tensor(state).to(self.device)
                action = self.choose_action(player, state)
                state_p, reward, battle_over, _ = player.step(action)
                sample = Sample(
                    state,
                    action,
                    reward,
                    state_p,
                    battle_over
                )
                self.store_sample(sample)
                steps += 1

                # if len(self.D) < self.memory_size:
                #     continue
                
                self.gradient_step()

                Q_hat_counter += 1
                if Q_hat_counter % self.reset_Q_steps == 0:
                    print('Copied Q')
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

            if ep % 10 == 0 and len(win_rate) > 0:
                print(f'Current win rate: {win_rate[-1]} ({player.n_won_battles}/{player.n_finished_battles})')
                print(f'Steps: {steps}')
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
            'losses': self.losses,
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
                state = torch.tensor(state).to(self.device)
                action = self.choose_action(player, state, epsilon=0)
                _, _, battle_over, _ = player.step(action)
            player.reset()

        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'n_battles': n_battles,
            'n_wins': n_wins
        }