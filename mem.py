import logging
import torch
import torch.nn as nn


KB = 1024
MB = 1024*KB
GB = 1024*MB


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def user_friendly(b: int) -> tuple[str, int]:
    mem = b
    mem_k = mem / KB
    mem_m = mem / MB
    mem_g = mem / GB
    if mem_g > 1:
        return ("G", mem_g)
    elif mem_m > 1:
        return ("M", mem_m)
    elif mem_k > 1:
        return ("K", mem_k)
    else:
        return ("b", mem)

def num_params_of(logger: logging.Logger, model: nn.Module) -> None:
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    (u, v) = user_friendly(mem)
    logger.info("Total model size: \t%2.1f%s", v, u)


def mem_usage(logger: logging.Logger, s: str) -> None : 
    logger.info("%s", s)
    (u, v) = user_friendly(torch.cuda.memory_allocated(0))
    logger.info("torch.cuda.memory_allocated:    \t%2.1f%s", v, u)
    (u, v) = user_friendly(torch.cuda.memory_reserved(0))
    logger.info("torch.cuda.memory_reserved:     \t%2.1f%s", v, u)
    (u, v) = user_friendly(torch.cuda.max_memory_reserved(0))
    logger.info("torch.cuda.max_memory_reserved: \t%2.1f%s", v, u)
