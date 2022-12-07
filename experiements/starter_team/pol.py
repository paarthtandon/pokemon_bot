import random
from env import showdown_to_switch, action_to_showdown

class EpsilonPolicy:
    def __init__(self, q, e):
        self.pol = q
        self.e = e

    def act(self, s, switches, only_switch=False):
        s = tuple(s)
        available_switches = showdown_to_switch(switches)
        available_actions = available_switches + [0, 1, 2, 3]

        qs = self.pol[s]
        if only_switch:
            mx = max([qs[a] for a in available_switches])
            actions = available_switches
        else:
            mx = max([qs[a] for a in available_actions])
            actions = available_actions
        A = []
        nA = []
        for a in actions:
            if qs[a] == mx:
                A.append(a)
            else:
                nA.append(a)
        An = len(A)
        try:
            op = ((1 - self.e) / An) + (self.e / 4)
            if random.random() >= (op * An) and len(nA) != 0:
                return action_to_showdown(
                    switches,
                    nA[random.randint(0, len(nA) - 1)]
                )
            return action_to_showdown(
                    switches,
                    A[random.randint(0, An - 1)]
                )
        except:
            print(A, nA, qs, mx)
            raise Exception()

class GreedyPolicy:
    def __init__(self, q):
        self.pol = q
    
    def act(self, s, switches, only_switch=False):
        s = tuple(s)
        available_switches = showdown_to_switch(switches)
        available_actions = available_switches + [0, 1, 2, 3]

        qs = self.pol[s]
        mx = float('-inf')
        mx_a = None
        if only_switch:
            actions = available_switches
        else:
            actions = available_actions
        for a in actions:
            if qs[a] > mx:
                mx_a = a
                mx = qs[a]
        return action_to_showdown(switches, mx_a)
            
