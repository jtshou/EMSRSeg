"""Random-state and data-position recovery for interrupted training."""
import random
import warnings
import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def resume_progress(state, rank, world_size, batch_size, workers):
    progress = state.get("training_progress")
    if progress is None:
        warnings.warn(
            "This legacy checkpoint has no data position or random states; "
            "model/optimizer state is restored, but its saved epoch restarts.")
        return None
    expected = (world_size, batch_size, workers)
    actual = tuple(progress[key] for key in ("world_size", "batch_size", "workers"))
    if actual != expected:
        raise ValueError(
            "To restore the data stream, keep world size, batch size and workers "
            "unchanged: saved %r, requested %r" % (actual, expected))
    if len(progress["ranks"]) != world_size:
        raise ValueError("Incomplete per-rank training state")
    return progress["ranks"][rank]


def training_iterator(loader, progress=None):
    """Replay worker loading, then restore the RNG state at the saved update."""
    epoch_rng = progress["epoch_rng"] if progress is not None else capture_rng_state()
    if progress is not None:
        restore_rng_state(epoch_rng)
    iterator = iter(loader)
    completed = progress["batches_completed"] if progress is not None else 0
    if completed < 0 or completed > len(loader):
        raise ValueError("Checkpoint data position is outside this epoch")
    for _ in range(completed):
        next(iterator)
    if progress is not None:
        restore_rng_state(progress["rng"])
    return iterator, completed, epoch_rng
