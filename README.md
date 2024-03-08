# multi-rl-mdintel_test

## Installation

Build the main environment via the following:

`conda env create --file environment.yml`

You will also need to install a separate package for ambertools as this environment conflicts with main environment.

`conda create --name ambertools -c conda-forge ambertools`

You will need a path to the /bin folder of this environment to perform simulation operations ( see configuration).

## Configuration
Once installed, you will need to define the location of executables and data needed by different tasks.

### Simulation Tasks
The simulation tasks use Amber.

Amber: Activate the "ambertools" environment then call:

dirname `which ambpdb`

This command will produce the path where the ambertools executables are found. Save that path in `multirl/config.py` as `AMBER_BIN_PATH`.

In md.yml, change `pdb_file` and `top_file` according to the files you need.
Included in this test are `2pwz_G.top` and `2pwz_G.pdb`. 

Finally, run the test via:

`python run_simulation.py`

You will get runtime information in `runtimes.json`

Provided is a jupyter notebook with instructions on evaluating performance: `evaluate-performance.ipynb`
