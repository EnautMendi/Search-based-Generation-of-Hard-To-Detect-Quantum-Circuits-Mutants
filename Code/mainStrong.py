
import argparse
import json
import os

import numpy as np
import pandas as pd
from jmetal.algorithm.multiobjective import RandomSearch
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.operator.crossover import SBXCrossover
from qiskit import QuantumCircuit

from scipy.stats import chisquare

from executeCircuit import circuitInitialization, execute_circuit, createInputs
from generateMutants import getPositionGate, createHighMutant
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion



def compareOutputs(oracle_output, mutant_output):
    result = False

    oracle_output = str(oracle_output)
    oracle_output = oracle_output.replace("'", "\"")
    mutant_output = str(mutant_output)
    mutant_output = mutant_output.replace("'", "\"")

    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)

    sorted_expected = dict(sorted(expected.items()))
    sorted_observed = dict(sorted(observed.items()))


    if len(list(sorted_observed.values())) == len(list(sorted_expected.values())):
        if sorted_expected.keys() == sorted_observed.keys():
            results = chisquare(list(sorted_observed.values()), list(sorted_expected.values()))
            if results[1] < 0.01:
                result = True
        else:
            result = True
    else:
        result = True


    return result
def compareResults(results_df):
    test_cases = 0
    for inp in inputs:
        oracle_ouput = oracle_df[oracle_df['input'] == str("'" + inp + "'")]
        mutant_output = results_df[results_df['input'] == str("'" + inp + "'")]
        killed = compareOutputs(oracle_ouput['counts'].values[0], mutant_output['counts'].values[0])
        if killed:
            test_cases = test_cases + 1

    return test_cases


def createInstructionsFromParams(variables):
    column_names = ['Name', 'Operator', 'Position', 'Gate', 'New_gate', 'Params', 'New_params', 'Qubits', 'New_qubits']
    df = pd.DataFrame(columns=column_names)
    for x in range(num_mutations):
        new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': round(variables[(4*x)]), 'Gate': 'Gap',
                           'New_gate': 'u',
                           'Params': [], 'New_params': variables[(4*x)+1:(4*x)+4], 'Qubits': (),
                           'New_qubits': (position_gates_df[position_gates_df['Position'] == round(variables[(4*x)])]['Qubits'].values[0][-1],)}
        new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
        df = pd.concat([df, new_df], ignore_index=True)

    return df


def mutantStrength(variables):
    global current_fitness
    instructions = createInstructionsFromParams(variables)
    high_mutant = createHighMutant(instructions, qc_origin)
    mutant = QuantumCircuit.from_qasm_str(high_mutant.qasm())

    columns = ['file', 'shots', 'input', 'counts']
    results_df = pd.DataFrame(columns=columns)
    for input in inputs:
        qc = circuitInitialization(mutant, input)
        results = execute_circuit(qc, shots)
        new_row = {'file': filename, 'shots': shots, 'input': ("'" + str(input) + "'"), 'counts': results}
        results_df = results_df.append(new_row, ignore_index=True)

    num_tc = compareResults(results_df)

    if num_tc == 0:
        fitness = 100
    else:
        fitness = num_tc

    current_fitness = fitness
    return fitness


def createUpperBounds():
    upper_bounds = []
    for x in range(num_mutations):
        upper_bounds.append(len(position_gates_df) - 1)
        upper_bounds.append(2 * np.pi)
        upper_bounds.append(2 * np.pi)
        upper_bounds.append(2 * np.pi)

    return upper_bounds


class strongMutants(FloatProblem):
    def __init__(self, number_of_variables: int, algorithm: str):
        super(strongMutants, self).__init__()
        self.algorithm = algorithm
        self.lower_bound = number_of_variables*[0]
        self.upper_bound = createUpperBounds()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Test Cases']
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.number_of_variables = number_of_variables

    def number_of_variables(self) -> int:
        return self.number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        solution.objectives[0] = mutantStrength(solution.variables)

        if self.algorithm == 'GA':
            log_file = open((str(dirname) + r"/logs/log_GA_" + str(run) + ".txt"), "a")
        else:
            log_file = open((str(dirname) + r"/logs/log_RS_" + str(run) + ".txt"), "a")
        log_file.write(str(solution.objectives[0]) + "\n")
        log_file.close()

        return solution

    def name(self) -> str:
        return 'Strong mutants'

class DoubleTermination(TerminationCriterion):
    def __init__(self, expected_fitness: int, max_evaluations: int):
        super(DoubleTermination, self).__init__()
        self.expected_fitness = expected_fitness
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def update(self, *args, **kwargs):
        self.evaluations = kwargs["EVALUATIONS"]

    @property
    def is_met(self):
        criteria = ((current_fitness == self.expected_fitness) or (self.evaluations >= self.max_evaluations))
        return criteria


def executeGA():

    problem = strongMutants(number_of_variables=(num_mutations*4), algorithm='GA')

    ga = GeneticAlgorithm(
        problem=problem,
        population_size=10,
        offspring_population_size=1,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables, 10.0),
        crossover=SBXCrossover(0.9, 5.0),
        termination_criterion=DoubleTermination(expected_fitness=strength, max_evaluations=100000),
    )

    ga.run()
    solutions = ga.solutions
    FUN_file = open((str(dirname) + r"/GA_Results/Fun_GA_" + str(run) + ".txt"), "a")
    VAR_file = open((str(dirname) + r"/GA_Results/Var_GA_" + str(run) + ".txt"), "a")

    for solution in solutions:
        FUN_file.write(str(solution.objectives[0]) + ' \n')
        for var in solution.variables:
            VAR_file.write(str(var) + ', ')
        VAR_file.write(' \n')

    FUN_file.close()
    VAR_file.close()

    log_file = open((str(dirname) + r"/logs/log_GA_" + str(run) + ".txt"), "a")
    log_file.write('Computing time: ' + str(ga.total_computing_time) + "\n")
    log_file.close()

    # ga_evaluations = ga.evaluations

    problem_RS = strongMutants(number_of_variables=(num_mutations*4), algorithm='RS')

    RS = RandomSearch(
        problem=problem_RS,
        termination_criterion=StoppingByEvaluations(max_evaluations=100000),
    )

    RS.run()
    solutions_RS = RS.get_result()
    FUN_file = open((str(dirname) + r"/randomSearch/Fun_RS_" + str(run) + ".txt"), "a")
    VAR_file = open((str(dirname) + r"/randomSearch/Var_RS_" + str(run) + ".txt"), "a")

    for solution in solutions_RS:
        FUN_file.write(str(solution.objectives[0]) + ' \n')
        for var in solution.variables:
            VAR_file.write(str(var) + ', ')
        VAR_file.write(' \n')

    FUN_file.close()
    VAR_file.close()

    log_file = open((str(dirname) + r"/logs/log_RS_" + str(run) + ".txt"), "a")
    log_file.write('Computing time: ' + str(ga.total_computing_time) + "\n")
    log_file.close()

if __name__ == '__main__':
    runs = 30
    current_fitness = 10000
    shots = 20000
    parser = argparse.ArgumentParser(description="Mutant selection with GA")
    parser.add_argument("origin_file", help="Original file")
    parser.add_argument("oracle_file", help="Oracle output of original file")
    parser.add_argument("num_mutations", help="Number of mutations to apply each time")
    parser.add_argument("algo_name", help="Algorithm Name")
    parser.add_argument("strength", help="Number of test cases to reach")
    args = parser.parse_args()
    origin_file = args.origin_file
    oracle_file = args.oracle_file
    num_mutations = int(args.num_mutations)
    algo_name = args.algo_name
    strength = int(args.strength)
    oracle_df = pd.read_csv(oracle_file)

    filename = origin_file.split('/')[-1]
    dirname = r"GA_Strong/" + str(algo_name) + "/" + filename.split('.')[0]
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(dirname + '/GA_Results', exist_ok=True)
    os.makedirs(dirname + '/logs', exist_ok=True)
    os.makedirs(dirname + '/randomSearch', exist_ok=True)

    qc_origin = QuantumCircuit.from_qasm_file(origin_file)
    position_gates_df = getPositionGate(qc_origin)
    inputs = createInputs(qc_origin.num_qubits)

    for run in range(runs):
        executeGA()

