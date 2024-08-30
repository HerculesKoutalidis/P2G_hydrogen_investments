# P2G_Paper_2_Hydrogen_Experiments
This is the repository for P2G experiments with end product hydrogen.
## Install Requirements
Begin by installing the packages needed, found inside the ```requirements.txt``` file, in your python environment. Run  ```pip install -r /path/to/requirements.txt``` to do so.

## How to run the experiments
Run ```python filename.py -v1 value1 -v2 value2``` ,  where ```value1``` and ```value2``` are the input H2 sale per kg values (e.g 3.1 and 3.2). Two esperiments are initiated and run in parallel this way.

## How to change model parameters
Follow the ```parameters_guide.xlsx```. Each sensitivity case has different model parameters. Follow this file to change the data in ```input_parameters_S2.1.csv ``` and  ```input_parameters_S2.2.csv ``` accordingly.

## How to change the simulation horizon
Go inside the model file near the beginning (row 26), and change ```simulation_years ``` accordingly (1,2,3,4,5).

## Results
After an experiment finishes, a csv file (also readable in MS word) is exported in the same dir as the experiment file (in the ```models``` folder), containing the results.
