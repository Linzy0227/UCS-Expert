"""Small helpers for optional torch.distributed training."""

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_average(tensor: torch.Tensor) -> torch.Tensor:
    return all_reduce_sum(tensor) / get_world_size()


def init_distributed_mode(args) -> None:
    """Initialize a process group from torchrun or SLURM variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print(
            "Distributed environment variables were not found; using one process."
        )
        args.distributed = False
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training currently requires CUDA/NCCL.")
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    if is_primary():
        print(
            f"Initialized distributed training with {args.world_size} processes."
        )
