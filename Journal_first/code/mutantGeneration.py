import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Instruction
from enum import Enum

# Define the Enum
class mutations(Enum):
    u_gate = 0
    p_gate = 1
    rx_gate = 2
    ry_gate = 3
    rz_gate = 4
    replace_with_u_gate = 5
    #cx_gate = 2

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

def generateInstructions(individual, position_gates_df):
    column_names = ['Name', 'Operator', 'Position', 'Gate', 'New_gate', 'Params', 'New_params', 'Qubits', 'New_qubits']
    df_instructions = pd.DataFrame(columns=column_names)
    for mutation in individual:
        match mutation[0]:
            case mutations.u_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'u',
                                   'Params': [], 'New_params': mutation[2:5], 'Qubits': (),
                                   'New_qubits': (position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[0][-1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case mutations.p_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'p',
                                   'Params': [], 'New_params': [mutation[2],], 'Qubits': (),
                                   'New_qubits': (
                                   position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[0][-1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case mutations.rx_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'rx',
                                   'Params': [], 'New_params': [mutation[2], ], 'Qubits': (),
                                   'New_qubits': (
                                       position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[
                                           0][-1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case mutations.ry_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'ry',
                                   'Params': [], 'New_params': [mutation[2], ], 'Qubits': (),
                                   'New_qubits': (
                                       position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[
                                           0][-1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case mutations.rz_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Add', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'rz',
                                   'Params': [], 'New_params': [mutation[2], ], 'Qubits': (),
                                   'New_qubits': (
                                       position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[
                                           0][-1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case mutations.replace_with_u_gate:
                new_instruction = {'Name': 'First_Mutant', 'Operator': 'Replace', 'Position': mutation[1], 'Gate': 'Gap',
                                   'New_gate': 'u',
                                   'Params': [], 'New_params': mutation[2:5], 'Qubits': (),
                                   'New_qubits': (
                                   position_gates_df[position_gates_df['Position'] == mutation[1]]['Qubits'].values[0][
                                       -1],)}
                new_df = pd.DataFrame.from_dict(new_instruction, orient='index').T
                df_instructions = pd.concat([df_instructions, new_df], ignore_index=True)
            case _:
                print('A enum that is not supported was selected')

    return df_instructions


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
