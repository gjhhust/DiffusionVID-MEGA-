import os

import torch
import torch.distributed as dist

from mega_core.utils import gpu_indices, ompi_size, ompi_rank, get_master_ip


def init_dist(launcher, args, backend='nccl'):
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, "torch")
    elif launcher == 'mpi':
        _init_dist_mpi(backend, args)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


# def _init_dist_pytorch(backend, args):
#     print("backend",backend)
#     os.environ['MASTER_PORT'] = args.master_port
#     torch.cuda.set_device(args.local_rank)
#     torch.distributed.init_process_group(
#         backend=backend, init_method="env://"
#     )

from torch import distributed as torch_dist
def _init_dist_pytorch(backend, init_backend='torch', **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    rank = int(os.environ['RANK'])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    if init_backend == 'torch':
        kwargs["rank"] = local_rank
        torch_dist.init_process_group(backend=backend, **kwargs)
    elif init_backend == 'deepspeed':
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, **kwargs)
    elif init_backend == 'colossalai':
        import colossalai
        colossalai.launch_from_torch(backend=backend, **kwargs)
    else:
        raise ValueError(
            'supported "init_backend" is "torch" or "deepspeed", '
            f'but got {init_backend}')



def _init_dist_mpi(backend, args):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    dist_url = 'tcp://' + get_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is{}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))
