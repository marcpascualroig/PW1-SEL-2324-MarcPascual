# RULES
SEL Prokect 1: Implementation of a rule based-algorithm: RULES algorithm
Author: Marc Pascual Roig

Course: Supervised and Experiential Learning

### Folders and files

- `Data`: contains 4 csv/xlsx datasets. 
- `Source`:  with the scripts `main.py`, `preprocessing.py`, `rules.py` and `requirements.txt`. 
- `Results`: contains the resulting rules and metrics (precision, coverage and recall) of every dataset (in a folder named as the dataset) in ".txt" format (these files are generated as output when executing the main.py). There are different files, with the metrics in the train and test sets for different configurations of the algorithm.

There are 6 ".txt" files for each dataset:
- metrics of the rules in the train set (only relevant rules)
- metrics of the rules in the train set (all the rules)
- metrics of the rules in the test set and results (with only relevant rules and rules applied sequentially)
- metrics of the rules in the test set and results (with only relevant rules and voting)
- metrics of the rules in the test set and results (with all rules and rules applied sequentially)
- metrics of the rules in the test set and results (with all rules and voting)


- `Documentation`: contains the report of the project. 
- A `README.md` file.

### Execute the code

1. Open the project
2. Create a virtual environment using Python `python -m venv myenv`
3. Open the virtual environment `source myenv/bin/activate` (or '\myenv\Scripts\Activate.ps1').
4. Install the required dependencies `pip install -r requirements.txt`
5. Choose the dataset for the execution (uncomment the line in the main file with the desired dataset).
6. Run the main file of the project.
7. Output: .txt files with rules and metrics (in the Source folder).
