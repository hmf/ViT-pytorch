import logging
import torch
import torch.nn as nn


KB = 1024
MB = 1024*KB
GB = 1024*MB


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def user_friendly_val(b: int) -> tuple[str, int]:
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

def user_friendly(logger: logging.Logger, s: str, b: int) -> None:
    (u, v) = user_friendly_val(b)
    logger.info("%s%2.1f%s", s, v, u)

def num_params_of(logger: logging.Logger, model: nn.Module) -> None:
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    (u, v) = user_friendly_val(mem)
    logger.info("Total model size: \t%2.1f%s", v, u)


def mem_usage(logger: logging.Logger, s: str, device: torch.device) -> None : 
    logger.info("%s", s)
    user_friendly(logger, "torch.cuda.memory_allocated:    \t", torch.cuda.memory_allocated(device))
    user_friendly(logger, "torch.cuda.memory_reserved:     \t", torch.cuda.memory_reserved(device))
    user_friendly(logger, "torch.cuda.max_memory_reserved: \t", torch.cuda.max_memory_reserved(device))
