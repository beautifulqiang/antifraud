"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Safe setup for single-machine or distributed training.
    If running on a single GPU/CPU, this will skip distributed setup.
    """
    import torch.distributed as dist

    # 若分布式已初始化，则直接返回
    if dist.is_initialized():
        print("[Info] Distributed process group already initialized.")
        return

    # 如果不是多进程环境，直接跳过分布式
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("[Info] No MPI or distributed environment detected, using single-process mode.")
        return

    try:
        backend = "gloo"  # CPU/GPU均可
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group(backend=backend, init_method="env://")
        print(f"[Info] Distributed initialized (rank={dist.get_rank()}, world_size={dist.get_world_size()}).")
    except Exception as e:
        print(f"[Warning] Failed to initialize distributed environment: {e}")
        print("[Info] Falling back to single-process mode.")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    # 单机运行的特判
    import torch.distributed as dist
    if not dist.is_initialized():
        # 单机时直接跳过
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
