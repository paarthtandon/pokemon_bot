import random
from env import showdown_to_switch, action_to_showdown
import ujson
from ast import literal_eval

def load_q(fname):
    with open(fname, 'r') as f:
        q = ujson.load(f)
        q = {literal_eval(k): {literal_eval(j): l for j, l in v.items()} for k, v in q.items()}
        return q

class EpsilonPolicy:
    def __init__(self, q, e):
        self.pol = q
        self.e = e

    def act(self, s, available_actions):
        s = tuple(s)

        qs = self.pol[s]
        mx = max([qs[a] for a in available_actions])
        A = []
        nA = []
        for a in available_actions:
            if qs[a] == mx:
                A.append(a)
            else:
                nA.append(a)
        An = len(A)
        try:
            op = ((1 - self.e) / An) + (self.e / 4)
            if random.random() >= (op * An) and len(nA) != 0:
                return nA[random.randint(0, len(nA) - 1)]
            return A[random.randint(0, An - 1)]
        except:
            print(A, nA, qs, mx)
            raise Exception()

class GreedyPolicy:
    def __init__(self, q):
        self.pol = q
    
    def act(self, s, available_actions):
        s = tuple(s)

        qs = self.pol[s]
        mx = float('-inf')
        mx_a = None
        for a in available_actions:
            if qs[a] > mx:
                mx_a = a
                mx = qs[a]
        return mx_a
            
