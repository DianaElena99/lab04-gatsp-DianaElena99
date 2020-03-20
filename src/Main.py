from src.GeneticAlgorithm import Chromosome, GA
from src.utils import *
from random import seed
from src.Repo import fitnessFx, Repo, plotAFunction
import matplotlib.pyplot as plt

class Main:

    def run(self):
        repo = Repo("easy.txt")
        a = 0
        b = repo.container['nrNoduri']
        seed(1)
        noDim = b
        xref = [[generateNewValue(a,b)] for _ in range(0, 1000)]
        xref.sort()
        yref = [fitnessFx(xi) for xi in xref]

        # initialise de GA parameters
        gaParam = {'popSize': 10, 'noGen': 5, 'pc': 0.8, 'pm': 0.1}
        # problem parameters
        problParam = {'min': 0, 'max': b, 'function': fitnessFx, 'noDim': noDim, 'noBits': 8}

        # store the best/average solution of each iteration (for a final plot used to anlyse the GA's convergence)
        allBestFitnesses = []
        allAvgFitnesses = []
        generations = []

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


            # logic alg
            ga.oneGeneration()
            # ga.oneGenerationElitism()
            # ga.oneGenerationSteadyState()

            bestChromo = ga.bestChromosome()
            print('Best solution in generation ' + str(g) + ' is: x = ' + str(bestChromo.repres) + ' f(x) = ' + str(
                bestChromo.fitness))

        plt.ioff()
        best, = plt.plot(generations, allBestFitnesses, 'ro', label='best')
        mean, = plt.plot(generations, allAvgFitnesses, 'bo', label='mean')
        plt.legend([best, (best, mean)], ['Best', 'Mean'])
        plt.show()



if __name__ == '__main__':
    main = Main()
    main.run()
