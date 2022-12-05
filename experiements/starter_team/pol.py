import random

class EpsilonPolicy:
    def __init__(self, q, e):
        self.pol = q
        self.e = e

    def act(self, s):
        s = tuple(s)
        qs = self.pol[s]
        mx = max(list(qs.values()))
        A = []
        nA = []
        for a in range(6):
            if qs[a] == mx:
                A.append(a)
            else:
                nA.append(a)
        An = len(A)
        try:
            op = ((1 - self.e) / An) + (self.e / 4)
        except:
            print(A, nA, qs, mx)
            raise Exception()
        if random.random() < (op * An):
            return A[random.randint(0, An - 1)]
        return nA[random.randint(0, len(nA) - 1)]

class GreedyPolicy:
    def __init__(self, q):
        self.pol = q
    
    def act(self, s):
        s = tuple(s)
        qs = self.pol[s]
        mx = float('-inf')
        mx_a = None
        for a in range(6):
            if qs[a] > mx:
                mx_a = a
                mx = qs[a]
        return mx_a
            
