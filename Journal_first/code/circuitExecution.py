import json

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicProvider
from qiskit.quantum_info import hellinger_distance
from scipy.stats import chisquare


def getExecResults(qc, shots, filename):
    inputs = createInputs(qc.num_qubits)
    columns = ['file', 'shots', 'input', 'counts']
    results_df = pd.DataFrame(columns=columns)
    for input in inputs:
        qc_init = circuitInitialization(qc, input)
        results = execute_circuit(qc_init, shots)
        new_row = {'file': filename, 'shots': shots, 'input': ("'" + str(input) + "'"), 'counts': results}
        new_df = pd.DataFrame.from_dict(new_row, orient='index').T
        results_df = pd.concat([results_df, new_df], ignore_index=True)

    return results_df

def execute_circuit(qc, shots):
    backend = BasicProvider().get_backend("basic_simulator")

    # Compile and run the Quantum circuit on a local simulator backend
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit, shots=shots, seed_simulator=42)
    result = job.result()
    counts = result.get_counts()

    return counts

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

def getHellinger(oracle_output, mutant_output):
    oracle_output = str(oracle_output)
    oracle_output = oracle_output.replace("'", "\"")
    mutant_output = str(mutant_output)
    mutant_output = mutant_output.replace("'", "\"")

    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)

    distance = hellinger_distance(expected, observed)

    return distance
def compareResults(oracle_df, mutant_df):
    inputs = oracle_df['input'].values
    test_cases = 0
    avg_result = 0
    for inp in inputs:
        oracle_ouput = oracle_df[oracle_df['input'] == str(inp)]
        mutant_output = mutant_df[mutant_df['input'] == str(inp)]
        killed = compareOutputs(oracle_ouput['counts'].values[0], mutant_output['counts'].values[0])
        if killed:
            test_cases = test_cases + 1

        result = getHellinger(oracle_ouput['counts'].values[0], mutant_output['counts'].values[0])
        avg_result = avg_result + result

    avg_result = avg_result / len(inputs)

    return test_cases, avg_result

def createInputs(QubitNum):
    inputs = ("",)
    x = 0
    while x < 2 ** QubitNum:
    #while x < 2 ** 5:
        binariInput = str(bin(x))
        binariInput = binariInput[2:len(binariInput)]
        if len(binariInput) < QubitNum:
            y = len(binariInput)
            tmp = ""
            while y < QubitNum:
                tmp = tmp + str(0)
                y = y + 1
            binariInput = tmp + binariInput
        inputs = inputs + (binariInput,)
        x = x + 1
        #FORCE ONLY ONE INPUT FOR TESTING
        #x = x + 2 ** QubitNum
    return inputs[1:len(inputs)]

def circuitInitialization(qc, input):
    initialization = QuantumCircuit(qc.num_qubits)
    x = 0
    for bit in input:
        if bit == '1':
            initialization.x(x)
        x = x + 1

    qc = initialization.compose(qc)
    return qc