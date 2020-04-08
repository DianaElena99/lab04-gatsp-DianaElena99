from cmath import sqrt
from random import seed
import matplotlib.pyplot as plt

from reinforced_ga.genetic_algorithm import generateARandomPermutation, GA

seed(1)


def readData(path):
    data = {}
    f = open(path)
    lines = f.readlines()
    data['noNodes'] = len(lines)
    mat = [[0 for _ in range(len(lines))] for _ in range(len(lines))]
    for i in range(len(lines)):
        for j in range(len(lines)):
            coords_1 = lines[i].split(' ')
            coords_2 = lines[j].split(' ')
            x1, y1 = int(coords_1[0]), int(coords_1[1])
            x2, y2 = int(coords_2[0]), int(coords_2[1])
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
            mat[i][j] = dist
    data['mat'] = mat
    return data

data = readData('berlin52.txt')

def plotAFunction(xref, yref, x, y, xoptimal, yoptimal, message):
    plt.plot(xref, yref, 'b-')
    plt.plot(x, y, 'ro', xoptimal, yoptimal, 'bo')
    plt.title(message)
    plt.show()
    plt.pause(0.9)
    plt.clf()

# plot the function to be optimised
noDim = 1
xref = [generateARandomPermutation(51) for _ in range(0, 1000)]
xref.sort()

def fitnessFx(chain):
    dist = 0
    for i in range(1,len(chain)):
        dist+= data['mat'][chain[i]][chain[i-1]]
    return dist


yref = [fitnessFx(xi) for xi in xref]
plt.ion()
plt.plot(xref, yref, 'b-')
plt.xlabel('x values')
plt.ylabel('y = f(x) values')
plt.show()



# initialise de GA parameters
gaParam = {'popSize': 250, 'noGen': 500, 'pc': 0.8, 'pm': 0.1}
# problem parameters
problParam = {'min': 0, 'max': 52, 'popSize': data['noNodes'],  'mat':data['mat'] , 'function': fitnessFx, 'noDim': noDim, 'noBits': 8}

# store the best/average solution of each iteration (for a final plot used to anlyse the GA's convergence)
allBestFitnesses = []
allAvgFitnesses = []
generations = []
bestFits = []

ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()

for g in range(gaParam['noGen']):
    # plotting preparation
    allPotentialSolutionsX = [c.repres for c in ga.population]
    allPotentialSolutionsY = [c.fitness for c in ga.population]
    bestSolX = ga.bestChromosome().repres
    bestSolY = ga.bestChromosome().fitness
    allBestFitnesses.append(bestSolY)
    allAvgFitnesses.append(sum(allPotentialSolutionsY) / len(allPotentialSolutionsY))
    generations.append(g)
    #plotAFunction(xref, yref, allPotentialSolutionsX, allPotentialSolutionsY, bestSolX, [bestSolY],
    #             'generation: ' + str(g))

    # logic alg
    ga.oneGeneration()
    #ga.oneGenerationElitism()
    #ga.oneGenerationSteadyState()

    bestChromo = ga.bestChromosome()
    bestFits.append([bestChromo, bestChromo.fitness])
    print('Best solution in generation ' + str(g) + ' is: x = ' + str(bestChromo.repres) + ' f(x) = ' + str(
        bestChromo.fitness))

lista = sorted(bestFits, key=lambda x: x[1], reverse=False)
print(lista[0])

plt.ioff()
plt.clf()
best, = plt.plot(generations, allBestFitnesses, 'r-', label='best')
mean, = plt.plot(generations, allAvgFitnesses, 'b-', label='mean')
plt.legend([best, (best, mean)], ['Best', 'Mean'])
plt.show()
