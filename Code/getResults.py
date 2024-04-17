import os

import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
import rpy2.robjects as robjects
r = robjects.r
r.library('effsize')
r['source']('A12.R')
test = robjects.globalenv['A12']


def find_smallest_value(filepath, max_evaluations):
    smallest = None
    line_num = 0
    evaluation = 0
    # Open the file and read line by line
    with open(filepath, 'r') as file:
        for line in file:
            line_num += 1
            # Convert the line to a numerical value (assuming it contains numbers)
            try:
                value = int(line.strip())
                # Update the smallest value if it's None or smaller than the current smallest value
                if smallest is None or value < smallest:
                    smallest = value
                    evaluation = line_num
            except ValueError:
                # Skip lines that cannot be converted to float
                pass
            if line_num > max_evaluations:
                break
    return smallest, evaluation


def evaluateFitness(ga_array, rs_array):
    # Perform Mann-Whitney U test
    statistic_mw, p_value_mw = mannwhitneyu(ga_array, rs_array)

    r_ga_array = robjects.FloatVector(ga_array)
    r_rs_array = robjects.FloatVector(rs_array)
    df_result_r = test(r_ga_array, r_rs_array)

    mag = str(df_result_r[2]).split("\n")[0].split()[-1]
    oeffect = float(str(df_result_r[3]).split()[-1])

    evaluation = {'P_ValueMW': p_value_mw, 'A12_Magnitude': mag, 'A12_effect': oeffect}

    return evaluation


if __name__ == '__main__':
    general_paths = [
        r'C:\Users\Enaut\PycharmProjects\highOrderMutants\EX3_SSBSE\StrongMutants_Results\strong_mutants\GA_Strong',
        r'C:\Users\Enaut\PycharmProjects\highOrderMutants\EX3_SSBSE\StrongMutants_Results\strong_mutants\GA_HighOrder2']
    column_names = ['Algorithm', 'P_valueMW', 'A12_Mag', 'A12_Eff']
    df_stats = pd.DataFrame(columns=column_names)
    for i, general_path in enumerate(general_paths):
        for directory in os.listdir(general_path):
            algorithm = directory.split('_')[0] + '_' + str(i+1)
            directory_path = os.path.join(general_path, directory)
            column_names = ['Run', 'Algorithm']
            df_ga = pd.DataFrame(columns=column_names)
            df_rs = pd.DataFrame(columns=column_names)
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if filename.startswith('log_GA'):
                        max_eval = 100000
                        fitness, evaluation = find_smallest_value(filepath, max_eval)
                        run = int(filename.split('_')[-1].replace('.txt', ''))
                        new_line = {'Run': run, 'Fitness_GA': fitness, 'Evaluation_GA': evaluation, 'Algorithm': algorithm}
                        new_df = pd.DataFrame.from_dict(new_line, orient='index').T
                        df_ga = pd.concat([df_ga, new_df], ignore_index=True)

                for filename in files:
                    filepath = os.path.join(root, filename)
                    if filename.startswith('log_RS'):
                        run = int(filename.split('_')[-1].replace('.txt', ''))
                        max_eval = int(df_ga['Evaluation_GA'].values.astype(np.float32).max())
                        fitness, evaluation = find_smallest_value(filepath, max_eval)
                        new_line = {'Run': run, 'Fitness_RS': fitness, 'Evaluation_RS': evaluation, 'Algorithm': algorithm}
                        new_df = pd.DataFrame.from_dict(new_line, orient='index').T
                        df_rs = pd.concat([df_rs, new_df], ignore_index=True)

            # Merge the DataFrames based on the run and algorithm column
            df_tmp = pd.merge(df_ga, df_rs, on=['Run', 'Algorithm'])
            ga_array = df_tmp['Fitness_GA'].values.astype(np.float16)
            rs_array = df_tmp['Fitness_RS'].values.astype(np.float16)
            evaluation = evaluateFitness(ga_array, rs_array)

            new_line = {'Algorithm': algorithm, 'P_valueMW': evaluation.get('P_ValueMW'), 'A12_Mag': evaluation.get('A12_Magnitude'), 'A12_Eff': evaluation.get('A12_effect'), 'GA_Mean': ga_array.mean(), 'GA_std': ga_array.std(),'GA_eval_mean':df_ga['Evaluation_GA'].values.astype(np.float32).mean(),'GA_eval_std':df_ga['Evaluation_GA'].values.astype(np.float32).std(), 'RS_Mean':rs_array.mean(), 'RS_std': rs_array.std(),'RS_eval_mean':df_rs['Evaluation_RS'].values.astype(np.float32).mean(),'RS_eval_std':df_rs['Evaluation_RS'].values.astype(np.float32).std()}
            new_df = pd.DataFrame.from_dict(new_line, orient='index').T
            df_stats = pd.concat([df_stats, new_df], ignore_index=True)

    #print(df_stats)
    df_stats.to_csv('results_cut_GA_max.csv', index=False)  # Set index=False to exclude row indices in the CSV file