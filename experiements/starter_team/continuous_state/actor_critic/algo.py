import torch
from torch import nn
from collections import deque
import random
from env import action_to_showdown, showdown_to_switch
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

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

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class AC:

    def __init__(
        self,
        memory_size=10000,
        minibatch_size=32,
        state_dim=8,
        action_dim=7,
        gamma=1,
        actor_lr=0.01,
        critic_lr=0.01,
        checkpoint_path=None
    ):
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.D = deque([], memory_size)
        self.gamma = gamma
        self.loss = nn.MSELoss()
        self.checkpoint_path = checkpoint_path
        self.actor_losses = []
        self.actor_loss_means = []
        self.critic_losses = []
        self.critic_loss_means = []

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_opt = torch.optim.SGD(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)
    
    def checkpoint(self, eps):
        torch.save(self.actor, f'{self.checkpoint_path}/model_a{eps}.pt')
        torch.save(self.critic, f'{self.checkpoint_path}/model_c{eps}.pt')
    
    def load_checkpoint(self, afname, cfname):
        pass
        self.actor = torch.load(afname)
        self.critic = torch.load(cfname)
    
    def store_sample(self, sample):
        self.D.append(sample)

    def sample_minibatch(self):
        return random.sample(list(self.D), min(len(list(self.D)), self.minibatch_size))

    def td_target(self, sample):
        state_p = torch.tensor(sample.state_p)
        reward = sample.reward

        state_v = self.critic(sample.state)
        state_p_v = self.critic(state_p)

        delta = reward + (self.gamma * state_p_v) - state_v

        return delta

    def update_actor(self, sample, target):
        logits = self.actor(sample.state)
        # print(logits)
        action_probs = torch.softmax(logits, 0, logits.dtype)
        # print(action_probs)
        action_prob = action_probs[sample.action]
        # print(action_prob)
        action_prob = torch.max(action_prob, torch.tensor(1e-4))
        if action_prob.item() == 0:
            print('Failed actor update')
            return
        loss = -torch.log(action_prob) * target
        # print(action_prob, target)
        #print(f'Actor Loss: {loss}')
        self.actor_losses.append(loss.item())
        self.actor_opt.zero_grad()
        loss.backward(retain_graph=True)
        # for name, param in self.actor.named_parameters():
        #     print(name, param.grad.norm())
        self.actor_opt.step()

    def update_critic(self, target):
        loss = torch.square(target)
        #print(f'Critic Loss: {loss}')
        self.critic_losses.append(loss.item())
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

    def choose_action(self, player, state):
        possible_moves = range(len(player.current_battle.available_moves))
        possible_switches = player.current_battle.available_switches
        possible_switch_actions = showdown_to_switch(possible_switches)
        possible_actions = list(possible_moves) + possible_switch_actions

        with torch.no_grad():
            #print(state)
            #print('\n\n')
            probs = self.actor(state)
            #print('\n\n')
            #print(probs)
            #print(possible_actions)
            probs = torch.tensor([probs[a].item() for a in possible_actions])
            probs = torch.softmax(probs, 0, probs.dtype)
            #print(probs)
            dist = torch.distributions.Categorical(probs=probs)
            action = possible_actions[dist.sample().item()]
            #print(action)
            while action not in possible_actions:
                action = dist.sample().item()
        
        return action, action_to_showdown(possible_switches, action)

    def train(self, player, episodes):
        self.actor.train()
        self.critic.train()
        rewards = []
        win_rate = []
        actions = {}
        eps = []
        cur_reward = 0
        steps = 0

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
                steps += 1
                
                target = self.td_target(sample)
                self.update_actor(sample, target)
                self.update_critic(target)
                
                cur_reward += reward

                try:
                    rewards.append(cur_reward)
                    if player.n_finished_battles != 0:
                        win_rate.append(
                            player.n_won_battles / player.n_finished_battles
                        )
                    else:
                        win_rate.append(0)
                    eps.append(ep)

                    self.actor_loss_means.append(sum(self.actor_losses) / len(self.actor_losses))
                    self.critic_loss_means.append(sum(self.critic_losses) / len(self.critic_losses))
                except:
                    print('Failed Step')
                    continue

                state = state_p

            if ep % 10 == 0 and len(win_rate) > 0:
                print(f'Current win rate: {win_rate[-1]} ({player.n_won_battles}/{player.n_finished_battles}) D: {len(self.D)}')
                action_p = sorted(
                    list(actions.items()),
                    key=lambda x: x[0]
                )
                action_p = [(a[0], round(a[1] / sum(actions.values()), 4)) for a in action_p]
                print(f'Steps: {steps}\nactions: {action_p}')
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
            'actor_losses': self.actor_loss_means,
            'critic_losses': self.critic_loss_means,
            'win_rate': win_rate,
            'n_battles': n_battles,
            'n_wins': n_wins
        }
    
    def evaluate(self, player, episodes):
        self.actor.eval()
        self.critic.eval()

        for ep in range(episodes):
            print(f'Running battle: {ep}')
            state, _, battle_over, _, = player.step(0)
            while not battle_over:
                state = torch.tensor(state)
                _, action_show = self.choose_action(player, state)
                state, _, battle_over, _ = player.step(action_show)
            player.reset()

        n_battles = player.n_finished_battles
        n_wins = player.n_won_battles
        
        return {
            'n_battles': n_battles,
            'n_wins': n_wins
        }
