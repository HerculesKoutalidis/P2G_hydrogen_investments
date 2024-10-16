#%%
import os, time, argparse, multiprocessing, numpy as np
from multiprocessing import Pool
# %%

def operation_function(value1, value2):
    print(f'Start processing {value1}')
    start_time = time.perf_counter()
    result = value1 ** value2
    time.sleep(value1)
    end_time = time.perf_counter()
    duration = round(end_time-start_time,4)
    print(f'Finished processing {value1}, result: {result}, duration: {duration}')

    

# %%
