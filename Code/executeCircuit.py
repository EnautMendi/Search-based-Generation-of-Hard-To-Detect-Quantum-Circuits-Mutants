
from qiskit import QuantumCircuit, BasicAer, execute
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def execute_circuit(qc, shots):
    backend = BasicAer.get_backend("qasm_simulator")

    # Compile and run the Quantum circuit on a local simulator backend
    job = execute(qc, backend, shots=shots, seed_simulator=42)
    result = job.result()
    counts = result.get_counts()

    return counts

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
