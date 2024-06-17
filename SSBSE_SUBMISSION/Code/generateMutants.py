
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Instruction


AllGates = ("x", "h", "p", "t", "s", "z", "y", "id", "rx", "ry", "rz", "sx", "swap", "rzz", "rxx", "cx", "cz", "ccx", "cswap", "cp", "u") #All gates that are implemented
OneQubit = ("x", "h", "p", "t", "s", "z", "y", "id", "rx", "ry", "rz", "sx", "u") #Gates that only affect one qubit
TwoQubit = ("swap", "rzz", "rxx", "cx", "cz", "cp") #Gates that affect two qubit
MoreThanTwoQubit = ("ccx", "cswap") #Gates that affect more than two qubit
NoParams = ("x", "h", "t", "s", "z", "y", "id", "sx", "swap", "cx", "cz", "ccx", "cswap") #All gates that don't need params
OneParam = ("p", "rx", "ry", "rz", "rzz", "rxx", "cp") #Gates that affect the phase and needs to specify a phase
TwoParam = () #Gates that affect the phase and needs to specify a phase
ThreeParam = ("u") #Gates that affect the phase and needs to specify a phase

def getPositionGate(qc):
    column_names = ['Position', 'Gate', 'Params', 'Qubits']
    df = pd.DataFrame(columns=column_names)
    x = 0
    for instruction in qc.data:
        if (instruction[0].name != 'measure') and (instruction[0].name != 'barrier'):
            new_line = {'Position': x, 'Gate': instruction[0].name, 'Params': instruction[0].params, 'Qubits': instruction.qubits}
            new_df = pd.DataFrame.from_dict(new_line, orient='index').T
            df = pd.concat([df, new_df], ignore_index=True)
            x = x + 1
    return df


def createGate(instruction):

    new_name = instruction.New_gate
    new_params = instruction.New_params
    new_qubits = instruction.New_qubits
    new_num_qubits = len(instruction.New_qubits)

    new_gate = CircuitInstruction(operation=Instruction(name=new_name,num_qubits=new_num_qubits,num_clbits=0,params=new_params),qubits=new_qubits)

    return new_gate

def createHighMutant(instructions, qc):
    add = False
    sorted_instructions = instructions.sort_values(by='Position')
    mutant = QuantumCircuit(qc.qubits, qc.clbits)
    mutated = 0
    for x, gate in enumerate(qc.data):
        if not sorted_instructions[sorted_instructions['Position'] == int(x)].empty:
            instructions_position = sorted_instructions[sorted_instructions['Position'] == int(x)]
            for ind, instruction in instructions_position.iterrows():
                if instruction.Operator == 'Add':
                    new_gate = createGate(instruction)
                    mutant.append(new_gate)
                    add = True
                elif instruction.Operator == 'Replace':
                    new_gate = createGate(instruction)
                    mutant.append(new_gate)
                mutated = mutated + 1
            if add:
                mutant.append(gate)
                add = False
        else:
            mutant.append(gate)

    while mutated < len(sorted_instructions): #NEED TO CHECK THE QUBIT NUMBER WHERE TO ADD AT THE END OF CIRCUIT
        mutated = mutated + 1
        new_gate = createGate(sorted_instructions.iloc[mutated])
        mutant.append(new_gate)  # NEED TO CHECK THE PARAMETERS, QUBITS AND VALUES
    return mutant


