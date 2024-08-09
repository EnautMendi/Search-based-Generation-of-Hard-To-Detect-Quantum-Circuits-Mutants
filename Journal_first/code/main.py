import argparse
import random
import time

import numpy as np
import pathlib

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

def init_individual(num_subarrays):
    return [init_subarray() for _ in range(num_subarrays)]

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
    elif num_tc == strength:
        fitness = num_tc
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
        if record['best_fit'] == strength:
            return True
    return False

# Define an algorithm function that logs each generation
def eaSimpleWithLogbook(log_filename, population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['best_fit']
    overall_best_fit = 10000000

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        population[:] = toolbox.select(offspring, len(population))

        if halloffame is not None:
            halloffame.update(population)
        record = stats.compile(population) if stats else {}
        new_best_fit = halloffame[0].fitness.values[0] if halloffame else None
        record['best_fit'] = new_best_fit
        if new_best_fit < overall_best_fit:
            overall_best_fit = new_best_fit
            best_gen = gen
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

        log_file = open(log_filename, "a")
        log_file.write(f"{gen},{len(offspring)},{record['min']},{record['max']},{record['best_fit']}\n")
        log_file.close()

        if customStoppingCriterion(logbook, strength):
            best_gen = gen
            print(f"Stopping criterion met at generation {gen} for GA")
            break

    return population, logbook, best_gen


# Random Search Algorithm with Hall of Fame
def random_search_with_hof(log_filename, pop_size, num_generations, verbose=__debug__):
    best_individual = None
    best_fitness = 10000000
    hall_of_fame = []

    if verbose:
        print("individual    fitness")

    population = [init_individual(num_mutations) for _ in range(pop_size*num_generations)]

    for i, individual in enumerate(population):
        fitness = evalIndividual(individual)
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_individual = individual
            best_eval = i

        # Update hall of fame
        if len(hall_of_fame) < pop_size:
            hall_of_fame.append((individual, fitness))
            hall_of_fame.sort(key=lambda x: x[1])
        elif fitness < hall_of_fame[-1][1]:
            hall_of_fame[-1] = (individual, fitness)
            hall_of_fame.sort(key=lambda x: x[1])

        if verbose:
            print(f"{i}     {fitness[0]}")

        log_file = open(log_filename, "a")
        log_file.write(f"{i},{fitness[0]}\n")
        log_file.close()

        if best_fitness == strength:
            best_eval = i
            print(f"Stopping criterion met at individual {i} for RS")
            break

    generation = int(best_eval/pop_size)
    return best_individual, best_fitness, hall_of_fame, generation

# Define a function to save the hall of fame to a file
def save_hall_of_fame(halloffame, filename):
    with open(filename, 'w') as f:
        f.write("Individual, ")
        for x in range(num_mutations):
            f.write(f"Mutation_{x+1},Position_{x+1},Params_{x+1},")
        f.write(f"Fitness\n")
        for i, individual in enumerate(halloffame):
            f.write(f"{i}, ")
            for subarray in individual:
                f.write(f"{subarray[0].name}, {subarray[1]}, {subarray[2:]}, ")
            f.write(f"{individual.fitness.values[0]}\n")

def save_hall_of_fame_RS(halloffame, filename):
    with open(filename, 'w') as f:
        f.write("Individual, ")
        for x in range(num_mutations):
            f.write(f"Mutation_{x+1},Position_{x+1},Params_{x+1},")
        f.write(f"Fitness\n")
        for i, individual in enumerate(halloffame):
            f.write(f"{i}, ")
            for subarray in individual[0]:
                f.write(f"{subarray[0].name}, {subarray[1]}, {subarray[2:]}, ")
            f.write(f"{individual[1]}\n")


# Step 6: Define the main function
def start():
    global num_mutations
    global position_gates_df
    global oracle_df
    global origin_qc
    global filename
    global shots
    global strength
    num_gen = 100
    pop_size = 100

    parser = argparse.ArgumentParser(description="Mutant generation with GA")
    parser.add_argument("origin_file", help="Original file")
    parser.add_argument("oracle_file", help="Oracle output of original file")
    parser.add_argument("num_mutations", help="Number of mutations to apply each time")
    parser.add_argument("strength", help="Number of test cases to reach")
    parser.add_argument("runs", help="Number of runs for each algorithm")
    args = parser.parse_args()

    origin_file = args.origin_file
    oracle_file = args.oracle_file
    num_mutations = int(args.num_mutations)
    strength = int(args.strength)
    runs = int(args.runs)

    oracle_df = pd.read_csv(oracle_file)
    origin_qc = QuantumCircuit.from_qasm_file(origin_file)
    position_gates_df = getPositionGate(origin_qc)
    filename = origin_file.split("\\")[-1]
    foldername = filename.replace('.qasm','')
    pathlib.Path(f'.\\results\\{foldername}').mkdir(parents=True,exist_ok=True)

    shots = 2 ** origin_qc.num_qubits * 2
    if shots < 1024:
        shots = 1024

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

    # Open a results file
    results_filename = f"results\\{foldername}\\all_results.txt"
    results_file = open(results_filename, "a")
    results_file.write("Run,GA_fitness,GA_generation,GA_time,RS_fitness,RS_generation,RS_time\n")
    results_file.close()

    for run in range(runs):
        # Create population
        population = toolbox.population(n=pop_size)

        # Add statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x) / len(x))
        stats.register("min", min)
        stats.register("max", max)

        # Add a hall of fame to store the best individuals
        hof = tools.HallOfFame(pop_size)

        # Add a logbook to track statistics
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields + ["best_fit"]

        # Open a log file
        log_filename_GA = f"results\\{foldername}\\evolution_log_GA_{run}.txt"
        log_file_GA = open(log_filename_GA, "a")
        log_file_GA.write("Generation,Evaluations,Avg,Min,Max,Best_Fit\n")
        log_file_GA.close()

        start_time_GA = time.time()
        # Run the algorithm
        population, logbook, generation_GA = eaSimpleWithLogbook(log_filename_GA, population, toolbox, cxpb=0.5, mutpb=0.5, ngen=num_gen, stats=stats,
                                                  halloffame=hof, verbose=True)
        end_time_GA = time.time()
        exec_time_GA = end_time_GA - start_time_GA

        best_ind = tools.selBest(population, 1)[0]
        #print(f"Best Individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")

        # Save the hall of fame to a file
        save_hall_of_fame(hof, f"results\\{foldername}\\hall_of_fame_GA_{run}.txt")

        # Open a log file
        log_filename_RS = f"results\\{foldername}\\evolution_log_RS_{run}.txt"
        log_file_RS = open(log_filename_RS, "a")
        log_file_RS.write("Generation,Evaluations,Best_Fit\n")
        log_file_RS.close()

        start_time_RS = time.time()
        best_individual_RS, best_fitness_RS, hall_of_fame_RS, generation_RS = random_search_with_hof(log_filename_RS, pop_size, num_gen, verbose=True)
        end_time_RS = time.time()
        exec_time_RS = end_time_RS - start_time_RS

        #print(f"Best Individual RS: {best_individual_RS}, Fitness: {best_fitness_RS}")

        # Save the hall of fame to a file
        save_hall_of_fame_RS(hall_of_fame_RS, f"results\\{foldername}\\hall_of_fame_RS_{run}.txt")

        results_file = open(results_filename, "a")
        results_file.write(f"{run},{best_ind.fitness.values[0]},{generation_GA},{exec_time_GA},{best_fitness_RS},{generation_RS},{exec_time_RS}\n")
        results_file.close()



if __name__ == '__main__':
    start()
