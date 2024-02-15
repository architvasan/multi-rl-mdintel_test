"""Test the placeholder functions"""
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from yaml import safe_load, dump

from md import batch_run_molecular_dynamics, run_molecular_dynamics

# Compare the runtimes for full-node and partitioned node CUDA
if __name__ == "__main__":
    # Path to example files
    pdb_file = Path('2pwz_G.pdb')
    yml_path = Path('md.yml')
    with yml_path.open() as fp:
        yml_config = safe_load(fp)
        
    # Load in the previous run data
    log_file = Path('runtimes.json')
    already_ran = set()
    if log_file.exists():
        already_ran = set(pd.read_json('runtimes.json', lines=True)[['pdb_file', 'sim_time', 'repeat']].apply(tuple, axis=1).to_list())

    for sim_time in [0.01, 0.1, 1, 10]:
        for repeat in range(2):
            if (pdb_file.name, sim_time, repeat) in already_ran:
                continue
        
            # Update the YAML file
            my_yaml = Path('my_md.yml')
            yml_config['sim_time'] = sim_time
            with my_yaml.open('w') as fo:
                dump(yml_config, fo)

            # Run a single molecular dynamics
            start_time = perf_counter()
            result = run_molecular_dynamics(pdb_file.read_text(), my_yaml.absolute())
            single_run = perf_counter() - start_time

            # Run a batch
            start_time = perf_counter()
            batch_run_molecular_dynamics([pdb_file.read_text()] * 4, my_yaml.absolute())
            multi_run = perf_counter() - start_time

            with log_file.open('a') as fp:
                print(json.dumps({
                    'pdb_file': pdb_file.name,
                    'sim_time': sim_time,
                    'repeat': repeat,
                    'single_time': single_run,
                    'multi_time': multi_run,
                }), file=fp)
