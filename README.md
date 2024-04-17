# Search-based strong mutant generation
## Code
The code used in the approach can be found under the [Code](Code/) folder. The following python scripts can be found:

- [mainStrong.py](Code/mainStrong.py) contains the main execution of the GA and RS.
- [generateMutants.py](Code/generateMutants.py) contains everything related to mutant generation.
- [executeCircuit.py](Code/executeCircuit.py) contains everything related to quantum circuit execution.
- [getResults.py](Code/getResults.py) Is the script used to check teh results and perform the statistical tests.

The code can be executed using teh following command:

    python mainStrong.py <origin_file> <oracle_file> <num_mutations> <algo_name> <strength>

As an example:
    
    python mainStrong.py origin_qc/ae_indep_qiskit_4.qasm oracles/ae_indep_qiskit_4.csv 2 ae 1

## Data
All the data obtained from teh executions can be obtained from the [Results](Results/) folder. 

- The folder contains raw execution data [GA_Strong](Results/GA_Strong) and [GA_HighOrder2](Results/GA_HighOrder2).
- Preprocessed data [firsOrder_data.csv](Results/firstOrder_data.csv) and [secondOrder_data.csv](Results/secondOrder_data.csv).
- Statistical tests results [stats_results.csv](Results/stats_results.csv).
