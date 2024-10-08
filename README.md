# Power to Gas Hydrogen end-product Experiments repository
Repository for Power to Gas experiments with end product hydrogen.
## Install Requirements
Install the packages needed to run the experiments, found inside the ```requirements.txt``` file, to your python environment. Connected to your environment, execute:  ```pip install -r /path/to/requirements.txt``` to do so.

## How to run the experiments
Execute: ```python filename.py -v1 value1 -v2 value2 -v3 value3 -v4 value4 -v5 value5 -y number_of_sim_years``` ,  where ```value1``` , ```value2``` etc. are the input H2 sale per kg values (e.g 3.1 and 3.2), and ```number_of_sim_years``` the number of simulation years of the investment (e.g. 1,2,..,5) . Five experiments are initiated, one for every H2 price, and run in parallel this way.
Advice: run your first set of experiments with number_of_sim_years = 1 , which takes ~20 min, to get acquainted with running the experiments.

## How to change model parameters
Models' parameters are in the ```input_parameters_S2.1.csv ``` and  ```input_parameters_S2.2.csv ``` files. Follow the ```parameters_guide.xlsx``` file to assign the correct parameters of each sensitivity scenario. Each sensitivity analysis case has different model parameters. Follow this file to change the data in ```input_parameters_S2.1.csv ``` and  ```input_parameters_S2.2.csv ``` accordingly.

## Results
After an experiment finishes, a csv file (also readable with text editors) is exported in the same directory as the executed experiment (.py) file (inside the respective ```models``` folder), containing the results. Another csv file is exported at the same dir, containing useful timeseries.
The names of both the results csv files are according to their scenario, simulated years, and sensitivity analysis scenario, to be easily recognizable.
