# Power to Gas Hydrogen end-product Experiments repository
Repository for Power to Gas experiments with end product hydrogen.
## Install Requirements
Install the packages needed to run the experiments, found inside the ```requirements.txt``` file, to your python environment. Connected to your environment, execute:  ```pip install -r /path/to/requirements.txt``` to do so.

## How to run the experiments
Execute: ```python filename.py -hpl <value1 value2 ..> -ss sensitivity_scenario -y number_of_sim_years``` ,  where value1, value2 etc. are the input H2 sale per kg prices (e.g 3.1 and 3.2),```sensitivity_scenario``` the sensitivity scenario name (a string: 'main",'LE1','LE2','LE3'), and ```number_of_sim_years``` the number of simulation years of the investment (e.g. 1,2,..,5) . One experiment for every H2 price is initiated, and run in parallel this way.
Advice: run your first set of experiments with number_of_sim_years = 1 , which takes ~10-30 min, to get acquainted with running the experiments.

## Results
After an experiment finishes, a csv file (also readable with text editors) is exported in the  ```Results``` folder (inside the respective ```models``` folder), containing the results. Another csv file is exported at the same dir, containing useful timeseries.
The names of both the results csv files are according to their scenario, simulated years, and sensitivity analysis scenario, to be easily recognizable.
