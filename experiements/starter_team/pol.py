import random

class EpsilonPolicy:
    def __init__(self, q, e):
        self.pol = q
        self.e = e

    def act(self, s, only_switch=False):
        s = tuple(s)
        qs = self.pol[s]
        if only_switch:
            mx = max([qs[a] for a in [4, 5]])
        else:
            mx = max([qs[a] for a in range(6)])
        A = []
        nA = []
        if only_switch:
            actions = [4, 5]
        else:
            actions = range(6)
        for a in actions:
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
    
    def act(self, s, only_switch=False):
        s = tuple(s)
        qs = self.pol[s]
        mx = float('-inf')
        mx_a = None
        if only_switch:
            actions = [4, 5]
        else:
            actions = range(6)
        for a in actions:
            if qs[a] > mx:
                mx_a = a
                mx = qs[a]
        return mx_a
            
