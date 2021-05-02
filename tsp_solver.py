import math
import argparse
import random
import heapq

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


# If you want to see the progress of the generations while performing the GA, set this to True
debug_print = True

# Default file paths used for testing

tsp_file = 'problems/rd400.tsp'



"""Default Hyper Parameters"""

population_size = 50
fitness_budget = 40000

# how big the archive for best solutions should be
elitism = 5

# tournament size for selection
k_size = 10

# Crossover
# How high the rate of ero crossovers should be, -1.0 for dynamic (the rest will be ox)
ero_rate = -1.0

# Mutation specifics
# How many 2opt swaps should be done when mutating
initial_mutation_strength = 400  # 400

# How many tries should be made to find suitable mutation candidates per attempted swap
mutation_tries = 400

# The rate for random city swaps after the 2opt mutation - currently unused
random_mutation_rate = 0.0

# Initialization - improvements to initial population

# Probability of nearest neighbor being applied to a randomly initialized tour
nn_rate = 0.0

# Probability of 2opt being applied to randomly initialized tour
twoopt_rate = 0.0

# Length of slice that the 2opt is being applied to (beware runtime!)
twoopt_slice_size = 50

# Budget for fine tuning via 2opt local at the end
fine_tuning_budget = 3000000

# Saving frequency of generations
generations_between_saving = 5


def log(string):
    if debug_print:
        print(string)


"""Functions to generate certain hyperparameters based on the input size"""


# finds a balance between convergence and execution time
# experimental scaling, the magic numbers are explained in the report
def ero_rate_normalizer(tour_len):
    if tour_len < 250:
        return 1.0
    elif 250 <= tour_len < 1500:
        return (-7.0 / 11000.0) * tour_len + (127.0 / 110.0)
    elif 1500 <= tour_len < 12000:
        return (-3.0 / 175000.0) * tour_len + (79.0 / 350.0)
    else:
        return 0.0


"""Supporting heuristics"""


def nn(perm):
    """
    Performs the nearest neighbor heuristic starting from the first element
    """
    nn_seed = list()
    nn_seed.append(perm.pop(1))
    while len(perm) != 0:
        nn_seed.append(perm.pop(
            perm.index(min(perm, key=lambda x: nn_seed[-1].distance_to(x)))))
    return nn_seed


def two_opt(perm, length, budget=-1):
    """
    (partially) performs the 2opt local search on a given permutation by iterating over
    all possible pairs until the budget run out or no more improvements can be made

    :param perm: The permutation to be optimized.
    :param length: How many elements should be included (starting from the beginning of perm)
    :param budget: How many swaps should be attempted at maximum, -1: no limit

    """

    done = False
    while not done:
        done = True
        for pos_a in range(length - 1):
            if not done:
                break
            for pos_c in range(pos_a + 2, length - 1):

                if budget == 0:
                    return
                else:
                    budget -= 1

                pos_b, pos_d = (pos_a + 1) % length, (pos_c + 1) % length

                if perm[pos_a].distance_to(perm[pos_b]) + perm[pos_c].distance_to(perm[pos_d]) > perm[
                    pos_a].distance_to(
                    perm[pos_c]) + perm[pos_d].distance_to(perm[pos_b]):
                    perm[pos_b:pos_d] = perm[pos_b:pos_d][::-1]

                    done = False
                    break


class Location:
    def __init__(self, id, x_coord, y_coord):
        self.id = id
        self.x = x_coord
        self.y = y_coord

    def __str__(self):
        return "{id: %d, (%f, %f)}" % (self.id, self.x, self.y)

    def distance_to(self, other):
        return math.sqrt(abs(other.x - self.x) ** 2 + abs(other.y - self.y) ** 2)


class Tour:
    """Tours are by design immutable -> fitness evaluation with every tour created. """

    # Assign global ids to individuals
    tour_counter = 0

    def __init__(self, location_list):
        self.perm = location_list
        self.dim = len(location_list)
        self.id = Tour.tour_counter
        Tour.tour_counter += 1

        self.fitness = 1.0 / self.distance()  # Inverse of distance as the fitness function

    def __str__(self):
        return "( tour_id: %d,\n%s )" % (self.id, "\n".join([str(l) for l in self.perm]))

    def distance(self):
        return sum([self.perm[i].distance_to(self.perm[i + 1]) for i in range(0, self.dim - 1)]) + self.perm[
            -1].distance_to(self.perm[0])

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join([str(l.id) for l in self.perm]))

    def visualize(self):
        locations = [(l.x, l.y) for l in self.perm]

        codes = [Path.MOVETO] + [Path.LINETO for i in range(self.dim - 2)] + [Path.CLOSEPOLY]

        path = Path(locations, codes)
        plt.close('all')
        fig, ax = plt.subplots()
        ax.add_patch(patches.PathPatch(path, facecolor='none', lw=1))
        ax.set_xlim(0, 1750)
        ax.set_ylim(0, 1750)
        plt.show()




class Genetic:

    def __init__(self, population, budget, elitism_rate, tournament_size, ero_rate,
                 mutation_strength, mutation_tries, random_mutation_rate,
                 nn_rate, twoopt_rate, twoopt_slice_size, seed_tour):

        """
        For explanations on the other parameters, see the hyperparameter section at the top

        :param seed_tour: The initial permutation to base all offspring on. Needs to be ordered.
        """

        self.population_size = population
        self.initial_budget = budget
        self.budget = budget
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size

        self.seed_tour = seed_tour  # ordered
        self.fittest_tour = seed_tour
        self.current_population = []
        self.current_generation = 0
        self.generation_scores = []

        self.tour_len = len(seed_tour.perm)

        if ero_rate == -1.0:
            self.ero_rate = ero_rate_normalizer(self.tour_len)
        else:
            self.ero_rate = ero_rate

        self.mutation_strength = mutation_strength

        self.mutation_tries = mutation_tries
        self.ran_mutation_rate = random_mutation_rate

        self.twoopt_slice = twoopt_slice_size
        self.twoopt_rate = twoopt_rate
        self.nn_rate = nn_rate

    def update_fittest(self, output):

        """
        checks to see if there is a better solution in the current population and replaces the old best solution
        optional output of the better solution
        """

        last_id = self.fittest_tour.id
        self.fittest_tour = max(self.current_population, key=lambda t: t.fitness)
        if output and self.fittest_tour.id != last_id:
            self.fittest_tour.visualize()
            log("Generation: %d , Distance: %d" % (self.current_generation, self.fittest_tour.distance()))

    def initialize(self):

        # As all members are automatically evaluated for fitness each time they are created,
        # the initialization has to be taken into account for the fitness budget
        if self.budget < self.population_size:
            self.population_size = self.budget
            self.budget = 0
        else:
            self.budget -= self.population_size

        population = list()
        perms = [self.seed_tour.perm[:] for i in range(0, population_size)]

        for perm in perms:
            random.shuffle(perm)

            # Each member has a chance (hyperparam) of getting fully optimized with nearest neighbor
            if random.random() < self.nn_rate:
                population.append(Tour(nn(perm)))

            # Each member has a chance (hyperparam) of getting a subpath of the length
            # "two_opt_sliced" optimized with 2-opt
            elif random.random() < self.twoopt_rate:
                pos_a = random.randint(0, self.tour_len - self.two_opt_slice - 1)
                pos_b = pos_a + self.two_opt_slice

                part1 = perm[:pos_a]
                part2 = perm[pos_a:pos_b]
                part3 = perm[pos_b:]

                two_opt(part2, self.two_opt_slice)

                population.append(Tour(part1 + part2 + part3))

            else:
                population.append(Tour(perm))

        return population

    def crossover_ox(self, perm1, perm2):
        """
        Ordered crossover - slices a random piece out of one of the parents and appends
        the other parent with all duplicate cities removed

        """

        position = sorted(random.sample(range(self.tour_len), 2))
        cut_out = perm1[position[0]:position[1]]

        return cut_out + [loc for loc in perm2 if loc not in cut_out]

    def crossover_ero(self, perm1, perm2):

        """
        Edge recombination crossover. Somewhat slow on very large inputs, use sparingly there

        """

        # Creating the map with all of the "neighboring" cities of each node across both parents,
        # the locations are stored in order of their id
        perm1_dict, perm2_dict = [None] * self.tour_len, [None] * self.tour_len

        for i in range(0, self.tour_len):
            a = (i + 1) % self.tour_len
            b = (i - 1) % self.tour_len

            perm1_dict[perm1[i].id - 1] = [perm1[a], perm1[b]]
            perm2_dict[perm2[i].id - 1] = [perm2[a], perm2[b]]

        joined_adj = [list(set(perm1_dict[i] + perm2_dict[i])) for i in range(0, self.tour_len)]

        k = []
        n = perm1[0]

        # Having the cities that are not in k as a shrinking list of integers (id) gives a slight performance increase
        # (vs looping through the entire parent permutation)
        missing_cities = list(range(1, self.tour_len + 1))

        while len(k) < self.tour_len:

            # Add the node as the next one in the order of the graph and
            # remove it from the list of potential neighbors of the other nodes
            k.append(n)
            missing_cities.remove(n.id)

            for l in joined_adj:
                if n in l:
                    l.remove(n)

            # If the last added node has neighbors, pick the one with the least neighbors and use it as the next node
            if joined_adj[n.id - 1]:
                first_candidate = min([[joined_adj[loc.id - 1], loc.id] for loc in joined_adj[n.id - 1]],
                                      key=lambda x: len(x[0]))

                candidates = [[joined_adj[loc.id - 1], loc.id] for loc in joined_adj[n.id - 1] if
                              len(joined_adj[loc.id - 1])
                              == len(first_candidate[0])]

                candidate = random.choice(candidates)
                n = self.seed_tour.perm[candidate[1] - 1]

            # If the last added node does not have neighbors, choose a random one as the next node
            elif len(k) < self.tour_len:
                n = self.seed_tour.perm[random.choice(missing_cities) - 1]

        return k

    # Currently unused
    def mutate_random(self, perm):
        """Swap mutation on the tours location sequence, swaps two random cities"""

        # Gradually decline mutation strength
        if self.budget / (self.initial_budget * 1.0) < self.mutation_strength / (self.initial_mutation_strength * 1.0):
            self.mutation_strength -= 1

        for i in range(self.mutation_strength):
            position1, position2 = random.sample(range(self.tour_len), 2)
            perm[position2], perm[position1] = perm[position1], perm[position2]
        return perm  # just for convenience, mutation happens inplace

    def mutate(self, perm):
        """
        2-opt-pseudo-mutation: Picks two random edges and swaps them if the swap would result in a reduced
        sum of the examined edge lengths. Repeats this until either the swap limit (hyperparam) is reached or
        it compares the maximum amount of edge pairs (hyperparam) without finding a suitable pair for swapping
        """

        swaps_left = self.mutation_strength

        while swaps_left > 0:

            # Do not perform further swaps if one round of attempts goes through without yielding any improving swaps
            temp = swaps_left
            swaps_left = 0
            for i in range(self.mutation_tries):

                pos_a, pos_c = random.randint(0, self.tour_len - 1), random.randint(0, self.tour_len - 1)

                while pos_a == pos_c:
                    pos_c = random.randint(0, self.tour_len - 1)

                if pos_a > pos_c:
                    pos_a, pos_c = pos_c, pos_a

                # Swap edges (reverse all nodes in between the two outer nodes) if the swap would be beneficial
                pos_b, pos_d = (pos_a + 1) % self.tour_len, (pos_c + 1) % self.tour_len
                if perm[pos_a].distance_to(perm[pos_b]) + perm[pos_c].distance_to(perm[pos_d]) > perm[
                    pos_a].distance_to(perm[pos_c]) + perm[pos_b].distance_to(perm[pos_d]):
                    perm[pos_b:pos_d] = perm[pos_b:pos_d][::-1]
                    swaps_left = temp - 1
                    break

        # Optional random node swaps to introduce forced diversity - currently unused
        if random.random() < self.ran_mutation_rate:
            position1, position2 = random.sample(range(self.tour_len), 2)
            perm[position2], perm[position1] = perm[position1], perm[position2]

        return perm  # just for convenience, mutation happens inplace

    def mate(self, crossover, tour1, tour2):
        if crossover == 1:
            perm = self.crossover_ero(tour1.perm, tour2.perm)
        else:
            perm = self.crossover_ox(tour1.perm, tour2.perm)

        return Tour(self.mutate(perm))

    def tournament(self):
        """Selection algorithm, selects k random members of the population """
        pool = random.sample(self.current_population, self.tournament_size)
        return max(pool, key=lambda t: t.fitness)

    def breed_tournament(self, target_size):
        """
        Creates a part of the next population with randomly distributed crossovers according
        to the ratio specified in the hyperparameters. Uses the exact ratio rather
        than a chance for each crossover because of the significant effect of variance on the runtime.
        """

        tokens = [1] * int(target_size * self.ero_rate)
        tokens = tokens + [0] * (target_size - len(tokens))

        random.shuffle(tokens)

        return [self.mate(token, self.tournament(), self.tournament()) for token in tokens]

    def next_population(self):
        if self.population_size < self.elitism_rate:
            self.elitism_rate = population_size

        # How many individuals need to be bred
        breeding_size = self.population_size - self.elitism_rate

        elite = heapq.nlargest(self.elitism_rate, self.current_population, key=lambda t: t.fitness)
        return elite + self.breed_tournament(breeding_size)

    def run(self):

        self.current_population = self.initialize()
        self.update_fittest(output=True)
        self.generation_scores.append(self.fittest_tour.distance())

        # While the elite does not warrant new fitness, I think it is fair to include them in
        # the fitness cost as the mutation requires quite a lot of comparisons,
        # evening the estimated cost of each generation out
        while self.budget > 0:

            if self.budget < self.population_size:
                self.population_size = self.budget
                self.budget = 0
            else:
                self.budget -= self.population_size

            self.current_generation += 1
            self.current_population = self.next_population()
            self.update_fittest(output=True)
            self.generation_scores.append(self.fittest_tour.distance())

            if self.current_generation % generations_between_saving == 0:
                self.fittest_tour.write_to_file('solution.csv')


        return self.generation_scores


# Handling of command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("tsp_file", nargs='?', default=tsp_file)

parser.add_argument('-p', dest='population_size', type=int, default=population_size)
parser.add_argument('-f', dest='fitness_budget', type=int, default=fitness_budget)
parser.add_argument('-e', dest='elitism', type=int, default=elitism)
parser.add_argument('-k', dest='k_size', type=int, default=k_size)
parser.add_argument('-er', dest='ero_rate', type=float, default=ero_rate)
parser.add_argument('-ms', dest='initial_mutation_strength', type=int, default=initial_mutation_strength)
parser.add_argument('-mt', dest='mutation_tries', type=int, default=mutation_tries)
parser.add_argument('-mrr', dest='random_mutation_rate', type=float, default=random_mutation_rate)
parser.add_argument('-nnr', dest='nn_rate', type=float, default=nn_rate)
parser.add_argument('-toptr', dest='twoopt_rate', type=float, default=twoopt_rate)
parser.add_argument('-toptl', dest='twoopt_slice_size', type=int, default=twoopt_slice_size)
parser.add_argument('-fb', dest='fine_tuning_budget', type=int, default=fine_tuning_budget)

args = parser.parse_args()

log(args)


# Read in the tsp file
with open(args.tsp_file, "r") as f:
    raw_data = [line.strip("\n").split()[:3] for line in f.readlines()[6:-1] if "EOF" not in line]
    location_list = [Location(int(datum[0]), float(datum[1]), float(datum[2])) for datum in raw_data]
    first_tour = Tour(location_list)

# Initialize and run GA

ga = Genetic(args.population_size, args.fitness_budget, args.elitism, args.k_size, args.ero_rate,
             args.initial_mutation_strength, args.mutation_tries, args.random_mutation_rate,
             args.nn_rate, args.twoopt_rate, args.twoopt_slice_size,
             first_tour)

ga.run()

# Fine tune the created tour by applying some 2-opt local search
two_opt(ga.fittest_tour.perm, len(ga.fittest_tour.perm), budget=args.fine_tuning_budget)

# Result Visualization
print(ga.fittest_tour.distance())
ga.fittest_tour.visualize()

ga.fittest_tour.write_to_file('solution.csv')






