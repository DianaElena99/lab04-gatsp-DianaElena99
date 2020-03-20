import random
from random import randrange
import matplotlib.pyplot as plt
from src.utils import generateNewValue


class Repo:
    def __init__(self, path):
        self.path = path
        self.container = self.LoadFromFile()

    def LoadFromFile(self):
        data = {}

        lines = open(self.path).readlines()

        nrNoduri = int(lines[0])
        data['nrNoduri'] = nrNoduri

        mat = []
        for i in range(1, nrNoduri + 1):
            mat.append([int(x) for x in lines[i].split(' ')])
        data['mat'] = mat

        return data


def fitnessFx(chain):
    count = 0
    chainn = chain[0]
    re = Repo("easy.txt")
    for i in range(1, len(chainn)):
        print(re.container['mat'][chainn[i-1]][chainn[i]])
        count += re.container['mat'][chainn[i-1]][chainn[i]]
    print("The value of fitness function for chain : ", chainn, " is ", count)
    return count

def plotAFunction(xref, yref, x, y, xoptimal, yoptimal, message):
    plt.plot(xref, yref, 'b-')
    plt.plot(x, y, 'ro', xoptimal, yoptimal, 'bo')
    plt.title(message)
    plt.show()
    plt.pause(0.9)
    plt.clf()

if __name__ == '__main__':
    repo = Repo("easy.txt")
    print(repo.container)
    chain = generateNewValue(0, repo.container['nrNoduri'])
    print(chain)
    fit = fitnessFx(chain)
    print(fit)

