import random


def generateNewValue(a,b):
    s = [a]
    rez = random.sample([y for y in range(1,b)], b-1)
    for i in rez:
        s.append(i)
    s.append(a)
    return s

