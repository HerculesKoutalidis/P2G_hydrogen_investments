# Power to Gas Hydrogen end-product Experiments repository
This is the repository for P2G experiments with end product hydrogen.
## Install Requirements
Begin by installing the packages needed run the experiments, found inside the ```requirements.txt``` file, to your python environment. Execute  ```pip install -r /path/to/requirements.txt``` to do so.

## How to run the experiments
Run ```python filename.py -v1 value1 -v2 value2 -v3 value3 -v4 value4 -y number_of_sim_years``` ,  where ```value1``` , ```value2``` etc. are the input H2 sale per kg values (e.g 3.1 and 3.2), and ```number_of_sim_years``` the number of simulation years of the investment (e.g. 1,2,..) . Four experiments are initiated and run in parallel this way.

## How to change model parameters
Follow the ```parameters_guide.xlsx```. Each sensitivity case has different model parameters. Follow this file to change the data in ```input_parameters_S2.1.csv ``` and  ```input_parameters_S2.2.csv ``` accordingly.

## Results
After an experiment finishes, a csv file (also readable with text editors) is exported in the same directory as the experiment file (the respective ```models``` folder), containing the results.
