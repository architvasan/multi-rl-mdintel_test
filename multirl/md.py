"""Functions related to running and analyzing molecular dynamic trajectories"""
from concurrent.futures import Future, as_completed, wait, ALL_COMPLETED
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import os

import numpy as np

from multirl.utils.pinning import get_persistent_process_pool, stop_ongoing_work, clear_persistent_pool
import multirl.config

logger = logging.getLogger(__name__)


def batch_run_molecular_dynamics(
        sequences: list[str],
        md_yml: Path,
        trailing_intolerance: int | None = None,
        trailing_impatience: float | None = None,
        n_gpus: int = 4
) -> tuple[list[np.ndarray], list[BaseException | None]]:
    """Run molecular dynamics on each member of a batch of sequences

    Args:
        sequences: Batch of sequences to be evaluated, each string is a different PDB file content
        md_yml: Path to the configuration file for the MD computation
        trailing_intolerance: How many trailing tasks to kill if they do not finish within `trailing_impatience` of the others
        trailing_impatience: Timeout before killing tasks which do not finish. Set to `None` to wait forever. (Units: s)
        n_gpus: Number of GPUs to use
    Returns:
        - RMSF for each sequence. ``None`` if the computation failed
        - Error message from the simulation, if it failed
    """
    # Determine the impatience settings
    if trailing_intolerance is not None and trailing_impatience is None:
        raise ValueError('You must set an intolerance if you set an impatience for trailing tasks')
    num_required_to_finish = len(sequences)  # Finish all if desired
    if trailing_intolerance is not None:
        num_required_to_finish -= trailing_intolerance

    # Get an executor
    exc = get_persistent_process_pool(n_gpus)

    # Submit all sequences
    futures: list[Future] = [
        exc.submit(run_molecular_dynamics, seq, md_yml)
        for seq in sequences
    ]

    # Wait for the required number of tasks to finish
    for _ in zip(as_completed(futures), range(num_required_to_finish)):
        pass

    # If we're impatient, kill process pool if others do not finish
    if num_required_to_finish != len(sequences):
        _, not_done = wait(futures, timeout=trailing_intolerance, return_when=ALL_COMPLETED)
        if len(not_done) > 0:
            logger.info(f'There are {len(not_done)} lingering tasks. Killing the whole executor')
            stop_ongoing_work()

    # Return all results, or stacktrace
    outputs = []
    exceptions = []
    for future in futures:
        # Log whether an exception occurred
        exception = future.exception()
        exceptions.append(exception)

        # Log the result, if successful
        if exception is not None:
            outputs.append(None)
            logger.warning(f'Task failed due to {exception}')
            clear_persistent_pool()
        else:
            outputs.append(future.result())

    return outputs, exceptions


def run_molecular_dynamics(pdb: str, md_yml: Path) -> np.ndarray:
    """Perform a molecular dynamics calculation on a sequence

    Args:
        pdb: Initial structure of sequence
        md_yml: Yaml file that stores MD simulation setups
    Returns:
        RMSF, as computed from simulation
    """
    from multirl.sim.run import sim_eval  # Import inside to delay importing OpenMM until _after_ CUDA is set

    assert pdb is not None
    assert md_yml.exists(), f'Cannot find {md_yml}'
    original_path = Path().absolute()
    with TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            # Save the PDB file contents to a temporary file
            path = Path('my.pdb')
            path.write_text(pdb)

            # Run the simulation and analysis
            return sim_eval(md_yml, path, amber_bin=config.AMBER_BIN_PATH)
        finally:
            os.chdir(original_path)
