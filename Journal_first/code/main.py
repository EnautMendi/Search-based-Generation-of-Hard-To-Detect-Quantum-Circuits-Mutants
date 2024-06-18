import argparse
import random
import numpy as np

import pandas as pd
from deap import base, creator, tools, algorithms
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

from circuitExecution import getExecResults, compareResults
from mutantGeneration import mutations, generateInstructions, createHighMutant, getPositionGate




# Define Initialization Functions
def init_mutation():
    return random.choice(list(mutations))

def init_position():
    return random.randint(0, (position_gates_df.shape[0]-1))

def init_params():
    return random.uniform(0.0, np.pi*2)

def init_subarray():
    return [init_mutation(), init_position(), init_params(), init_params(), init_params()]

# Define the evaluation function
def evalIndividual(individual):
    instructions = generateInstructions(individual, position_gates_df)
    mutant = createHighMutant(instructions, origin_qc)
    mutant_str = dumps(mutant)
    qc_mut = QuantumCircuit.from_qasm_str(mutant_str)
    mutant_exec_df = getExecResults(qc_mut, shots, filename)

    num_tc, avg_dist = compareResults(oracle_df, mutant_exec_df)

    if num_tc == 0:
        fitness = 10000000
    else:
        fitness = num_tc + avg_dist

    return fitness,

# Custom mutation for subarrays
def mutate_subarray(individual, indpb):
    for subarray in individual:
        if random.random() < indpb:
            subarray[0] = init_mutation()
        if random.random() < indpb:
            subarray[1] = init_position()
        for i in range(2, len(subarray)):
            if random.random() < indpb:
                subarray[i] = init_params()
    return individual,

# Define a custom stopping criterion
def customStoppingCriterion(logbook, strength):
    for record in logbook:
        if record['best_fit'] <= strength:
            return True
    return False

# Define an algorithm function that logs each generation
def eaSimpleWithLogbook(log_file, strength, population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['best_fit']

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        population[:] = toolbox.select(offspring, len(population))

        if halloffame is not None:
            halloffame.update(population)
        record = stats.compile(population) if stats else {}
        record['best_fit'] = halloffame[0].fitness.values[0] if halloffame else None
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)
        log_file.write(f"{gen},{len(offspring)},{record['min']},{record['max']},{record['best_fit']}\n")

        if customStoppingCriterion(logbook, strength):
            print(f"Stopping criterion met at generation {gen}")
            break

    return population, logbook

# Define a function to save the hall of fame to a file
def save_hall_of_fame(halloffame, filename):
    with open(filename, 'w') as f:
        for i, individual in enumerate(halloffame):
            f.write(f"Individual {i}:\n")
            for subarray in individual:
                f.write(f"Enum: {subarray[0].name}, Floats: {subarray[1:]}\n")
            f.write(f"Fitness: {individual.fitness.values[0]}\n\n")


# Step 6: Define the main function
def start():
    global num_mutations
    global position_gates_df
    global oracle_df
    global origin_qc
    global filename
    global shots
    shots = 1024

    parser = argparse.ArgumentParser(description="Mutant generation with GA")
    parser.add_argument("origin_file", help="Original file")
    parser.add_argument("oracle_file", help="Oracle output of original file")
    parser.add_argument("num_mutations", help="Number of mutations to apply each time")
    parser.add_argument("strength", help="Number of test cases to reach")
    args = parser.parse_args()

    origin_file = args.origin_file
    oracle_file = args.oracle_file
    num_mutations = int(args.num_mutations)
    strength = int(args.strength)

    oracle_df = pd.read_csv(oracle_file)
    origin_qc = QuantumCircuit.from_qasm_file(origin_file)
    position_gates_df = getPositionGate(origin_qc)
    filename = origin_file.split('/')[-1]

    # dirname = r"GA_Strong/" + str(algo_name) + "/" + filename.split('.')[0]
    # os.makedirs(dirname, exist_ok=True)
    # os.makedirs(dirname + '/GA_Results', exist_ok=True)
    # os.makedirs(dirname + '/logs', exist_ok=True)
    # os.makedirs(dirname + '/randomSearch', exist_ok=True)

    # Create Custom Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Register Functions in Toolbox
    toolbox = base.Toolbox()

    toolbox.register("attr_subarray", init_subarray)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_subarray,
                     n=num_mutations)  # 2 subarrays = 2 mutations in one circuit
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    toolbox.register("evaluate", evalIndividual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Register custom mutation
    toolbox.register("mutate", mutate_subarray, indpb=0.2)

    population = toolbox.population(n=100)

    # Add a hall of fame to store the best individuals
    hof = tools.HallOfFame(100)

    # Add statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)
    stats.register("max", max)

    # Add a logbook to track statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields + ["best_fit"]

    # Open a log file
    log_file = open("evolution_log.txt", "a")
    log_file.write("Generation,Evaluations,Avg,Min,Max,Best_Fit\n")

    # Run the algorithm
    population, logbook = eaSimpleWithLogbook(log_file, strength, population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                                              halloffame=hof, verbose=True)
    log_file.close()

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best Individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")

    # Save the hall of fame to a file
    save_hall_of_fame(hof, "hall_of_fame.txt")



if __name__ == '__main__':
    start()
