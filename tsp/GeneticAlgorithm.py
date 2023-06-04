import random
from TSPData import TSPData
import numpy as np
import matplotlib.pyplot as plt

# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    # @param pCrossOver the chance of cross-over during reproduction
    # @param pMutation the chance of mutation during reproduction
    def __init__(self, generations, pop_size, p_cross_over=0.7, p_mutation=0.01):
        self.generations = generations
        self.pop_size = pop_size
        self.p_cross_over = p_cross_over
        self.p_mutation = p_mutation

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @param plot, normally False but can be set to True just to look at how the model is training.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data, plot=False):
        # Initialize the population
        population = self.initialize_population()
        best_fitness = -float("inf")
        best_population = []
        if plot:
            # Arrays used for plotting
            bps = np.empty(self.generations)
            avg_bps = np.empty(self.generations)
        # For each generation
        for i in range(self.generations):
            # Calculate the cumulative fitness ratio and fitness of each population member
            cumulative, fitness = self.fitness(population, tsp_data)
            # Check whether a better solution than the current best solution has been found.
            if (np.max(fitness) > best_fitness):
                best_population = population[np.argmax(fitness)]
                best_fitness = np.max(fitness)
            if plot:
                bps[i] = best_fitness
                avg_bps[i] = np.average(fitness)
            # Create a new population using the reproduction function.
            population = self.reproduction(population=population, cumulative=cumulative, p_cross_over=self.p_cross_over, p_mutation=self.p_mutation)
        # Calculate the fitness and check for a more optimal solution one last time.
        cumulative, fitness = self.fitness(population, tsp_data)
        if (np.max(fitness) > best_fitness):
            best_fitness = np.max(fitness)
            best_population = population[np.argmax(fitness)]
        if plot:
            bps[np.size(bps) - 1] = best_fitness
            avg_bps[np.size(avg_bps) - 1] = np.average(fitness)
            plt.plot(bps)
            plt.plot(avg_bps)
            plt.show()
            plt.plot(np.sqrt(np.divide(1.0, bps)))
            plt.plot(np.sqrt(np.divide(1.0, avg_bps)))
            plt.show()
        return best_population
    
    # Initializes the population, by making each chromosome the range [0, 17] and then shuffling the values.
    def initialize_population(self):
        population = np.zeros((self.pop_size, 18), dtype=np.int8)
        for i in range(self.pop_size):
            population[i] = np.arange(18)
            np.random.shuffle(population[i])
        return population
    
    # Calculates the fitness and cumulative fitness ratio of each population member
    def fitness(self, population, tsp_data):
        #total_distances = np.empty(self.pop_size)
        fitness_lambda = lambda x : self.fitness_helper(np.array(x), tsp_data)
        # Fitness is equal to 1e6 / (route_distance)^2
        total_distances = np.divide(1.0, np.square(np.fromiter(map(fitness_lambda, population), dtype=float)))
        fitness_ratio = np.multiply(100, np.divide(total_distances, np.sum(total_distances)))
        cumulative = np.cumsum(fitness_ratio)
        return cumulative, total_distances
    
    # Helper function which calculates the total distance of a route
    def fitness_helper(self, p, tsp_data):
        total_distance = tsp_data.start_distances[p[0]]
        total_distance = total_distance + np.sum(np.fromiter(map(lambda x: tsp_data.distances[p[x]][p[x + 1]], np.arange(17)), dtype=float))
        total_distance = total_distance + tsp_data.end_distances[p[17]]
        return total_distance
    
    # Function which returns a new population based on the current population,
    # it's cumulative fitness ratio, the chance of cross-over and the chance of mutation
    def reproduction(self, population, cumulative, p_cross_over, p_mutation):
        new_population = np.empty((self.pop_size, 18), dtype=np.int8)
        for i in range(int(self.pop_size / 2)):
            # First we do selection
            pop_numbers = np.multiply(100, np.random.uniform(size=2))
            parents = np.empty((2, 18), dtype=np.int8)
            for j in range(2):
                for k in range(len(cumulative)):
                    if (pop_numbers[j] < cumulative[k] or k == len(cumulative) - 1):
                        parents[j] = population[k]
                        break
            # Now we do either cross-over or cloning
            child_one = parents[0]
            child_two = parents[1]
            if (np.random.random() < p_cross_over):
                # Cross over in such a way that the order gets preserved while also keeping the constraint of no duplicate values in the chromosome
                cross_over_point = np.random.randint(0, 18)
                # We check what values are still missing from the slice taken from one parent and get the order of those values from the other parent
                needed_values = np.setdiff1d(np.arange(18), parents[0][:cross_over_point])
                child_one_build = parents[1][np.where(np.isin(parents[1], needed_values))]
                needed_values = np.setdiff1d(np.arange(18), parents[1][cross_over_point:])
                child_two_build = parents[0][np.where(np.isin(parents[0], needed_values))]
                child_one = np.concatenate((parents[0][:cross_over_point], child_one_build))
                child_two = np.concatenate((parents[1][cross_over_point:], child_two_build))
            # Now we check for mutation
            # We have two different mutations, one which randomly swaps two values and one which rotates all values.
            if (np.random.random() < p_mutation):
                if (np.random.random() < 0.5):
                    swaps = np.random.randint(0, 18, size=2)
                    child_one[swaps[0]], child_one[swaps[1]] = child_one[swaps[1]], child_one[swaps[0]]
                else:
                    child_one = np.roll(child_one, 1)
            if (np.random.random() < p_mutation):
                if (np.random.random() < 0.5):
                    swaps = np.random.randint(0, 18, size=2)
                    child_two[swaps[0]], child_two[swaps[1]] = child_two[swaps[1]], child_two[swaps[0]]
                else:
                    child_two = np.roll(child_two, 1)
            new_population[i * 2] = child_one
            new_population[i * 2 + 1] = child_two
        return new_population
            

            


                    