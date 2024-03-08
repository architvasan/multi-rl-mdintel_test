import os
import shutil
import numpy as np

from multirl.sim.amber import AMBER_param
from multirl.sim.sim import Simulate
from multirl.sim.utils import dict_from_yaml, cal_rmsf


def sim_eval(yml_file, pdb=None, amber_bin: str = '') -> np.ndarray:
    """Perform a simulation according to a YML specification

    Args:
        yml_file: Path to the YML file describing the simulation
        pdb: Path to the PDB file being run (optional)
        amber_bin: Path the directory holding amber binaries
    Returns:
         The RMSF from the computation
    """
    args = dict_from_yaml(yml_file)
    if not pdb:
        pdb = args['pdb_file']
    pdb, top = param(pdb, amber_bin=amber_bin)

    args['pdb_file'] = pdb
    args['top_file'] = top
    sim_path = sim(args)
    dcd = f"{sim_path}/output.dcd"

    rmsf = cal_rmsf(top, dcd)
    np.save(f'{sim_path}/rmsf.npy', rmsf)
    return rmsf


def param(pdb, **kwargs):
    """Prepare a run given the PDB file

    Args:
        pdb: Path to the PDB file being evaluated
    Returns:
        - Path to the PDB
        - Path to the topology file
    """
    host_dir = os.getcwd()

    # label for ligand identity
    pdb_code = os.path.basename(pdb)[:-4]

    # make the work dir
    work_dir = os.path.abspath(os.path.join(host_dir, 'input_' + pdb_code))
    os.makedirs(work_dir, exist_ok=True)

    # make a copy of pdb in the new dir
    pdb_copy = os.path.join(work_dir, os.path.basename(pdb))
    shutil.copy2(pdb, pdb_copy)

    # run and get the parameters
    try:
        amberP = AMBER_param(pdb_copy, forcefield='ff14SB',
                             watermodel='tip3p', **kwargs)
        pdb, top = amberP.param_comp()
    finally:
        os.chdir(host_dir)
    return pdb, top


def sim(args):
    """Perform the simulation given run configuration"""
    sim = Simulate(**args)
    return sim.md_run()
