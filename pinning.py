"""Utilities associated with intranode parallelism via ProcessPoolExecutor"""
import os
import logging
import signal
from time import sleep
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, get_context

logger = logging.getLogger(__name__)

# Global process pools
_ppex_pids: list[int] | None = None
_ppex_size: tuple[int, int] | None = None
_ppex: ProcessPoolExecutor | None = None


# Tool to get the pid of subprocesses (using a wait to ensure they block)
def _sleep_then_pid():
    sleep(0.1)
    return os.getpid()


def _pin_to_resources(rank_queue: Queue, total_ranks: int):
    """Pop the rank off the list queue to get your rank, then assign to associated resources """
    # Get my rank
    rank = rank_queue.get()

    # Add the environment variable
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        logger.info(f"Pinned to GPU: {rank}")
    else:
        logger.info("CUDA_VISIBLE_DEVICES is already set")
        return  # Don't bother with the CPU affinity either

    # Set the CPU affinity
    avail_cores = sorted(os.sched_getaffinity(0))  # Get the available processors
    cores_per_worker = len(avail_cores) // total_ranks
    assert cores_per_worker > 0, f"Affinity does not work if there are more workers than cores. Avail: {len(avail_cores)}. Workers: {total_ranks}"

    my_cores = avail_cores[cores_per_worker * rank:cores_per_worker * (rank + 1)]
    os.sched_setaffinity(0, my_cores)

    logger.info(f'Affixed rank {rank} to GPUs and CPUs')


def make_pinned_process_pool(n_gpus: int, tasks_per_gpu: int = 1) -> ProcessPoolExecutor:
    """Make a process pool executor where each worker is pinned to different GPU/CPUs

    Args:
        n_gpus: Number of GPUs on this system
        tasks_per_gpu: Number of workers pinned to each GPU (you should use MPS for >1)
    Returns:
        Process pool executor where each worker is pinned to a different set of resources
    """

    # Start by defining a queue
    context = get_context('spawn')

    # Make an MP queue to use in the startup function
    logger.info(f'Making a process pool with {n_gpus} workers')
    queue = context.Queue()
    for _ in range(tasks_per_gpu):
        for i in range(n_gpus):
            queue.put(i)

    # Launch the pool
    return ProcessPoolExecutor(max_workers=n_gpus * tasks_per_gpu, mp_context=context, initializer=_pin_to_resources, initargs=(queue, n_gpus * tasks_per_gpu))


def get_persistent_process_pool(n_gpus: int, tasks_per_gpu: int = 1) -> ProcessPoolExecutor:
    """Make or access a process pool where each worker is pinned to a different GPU

    Args:
        n_gpus: Number of GPUs on this system
        tasks_per_gpu: Number of workers pinned to each GPU (you should use MPS for >1)
    Returns:
        Process pool executor where each worker is pinned to a different set of resources
    """

    global _ppex, _ppex_size, _ppex_pids
    if _ppex is not None:
        # Best case: Exists and matches current size
        if (n_gpus, tasks_per_gpu) == _ppex_size:
            return _ppex
        else:
            logger.info(f'New pool is a different size ({n_gpus}) that what exists ({_ppex_size}. Shutting current one down')
            clear_persistent_pool()

    # Make the process pool, then store information about it
    _ppex = make_pinned_process_pool(n_gpus, tasks_per_gpu)
    _ppex_size = (n_gpus, tasks_per_gpu)
    _ppex_pids = []
    for future in [_ppex.submit(_sleep_then_pid) for _ in range(n_gpus * tasks_per_gpu)]:
        _ppex_pids.append(future.result())

    return _ppex


def stop_ongoing_work():
    """Stops all ongoing work from the pinned executor and then shuts it down

    Use this function if, for example, there are trailing tasks on the executor that you wish to halt
    """
    global _ppex_pids
    for pid in _ppex_pids:
        os.kill(pid, signal.SIGKILL)
    clear_persistent_pool()


def clear_persistent_pool():
    """Shut the persistent process pool down"""
    global _ppex, _ppex_size

    if _ppex is not None:
        logger.info(f'Shutting process pool down')
        _ppex.shutdown()
        _ppex = None
        _ppex_size = None
