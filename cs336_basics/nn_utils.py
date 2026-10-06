import torch 
from torch import nn 
from cs336_basics import nn_block
#from nn_block import softmax # 注意这里的import
from torch.nn import functional as F 
from collections.abc import Callable, Iterable
from typing import Optional 
import math 
import numpy as np 
import random

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs_max = torch.max(inputs, dim=-1, keepdim=True)[0]
    inputs_shift = inputs - inputs_max 
    inputs_norm = torch.exp(inputs_shift).sum(dim = -1, keepdim=True) 
    log_softmax = inputs_shift - torch.log(inputs_norm) # 注意这里的第一项用inputs_shift
    output = - 1 * log_softmax[torch.arange(inputs.shape[0]), targets] 
    return torch.mean(output) 


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-9, weight_decay=1e-4):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lmd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue 
                grad = p.grad.data 
                theta = p.data
                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = torch.zeros_like(grad) 
                m, v = state['m'], state['v']
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                lr_t = lr * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)) 
                theta.add_(theta, alpha = -lr * lmd) # 正则项修正,注意这里的参数alpha
                m.mul_(beta1).add_((1 - beta1) * grad)
                v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
                theta.add_(-lr_t * m / (v ** 0.5 + eps))
                state["t"] = t + 1  # Increment iteration number.
        return loss 


def cosine_annealing_schedule(t: int,
                              t_w: int,
                              t_c: int,
                              a_min: float,
                              a_max: float):
    if t < t_w:
        return  t / t_w * a_max 
    if t >= t_w and t < t_c:
        return a_min + 0.5 * ( 1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (a_max - a_min) 
    if t >= t_c:
        return a_min


def l2_norm_clip(parameters, max_l2_norm, epslion=1e-6):
    all_grad = torch.cat([param.grad.flatten() for param in parameters if param.grad is not None])  # 注意这里是对grad作norm 注意grad可能为空
    l2_norm = torch.sqrt(torch.sum(all_grad ** 2)).data 
    if l2_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(max_l2_norm / (l2_norm + epslion))


def data_load(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = x.shape[0]
    choice_end = data_len - context_length - 1
    in_start = [random.randint(0, choice_end) for i in range(batch_size)] 
    nt_start = [i + 1 for i in in_start]
    in_batch = torch.stack([torch.Tensor(x[start : start + context_length]) for start in in_start], dim = 0) 
    nt_batch = torch.stack([torch.Tensor(x[start : start + context_length]) for start in nt_start], dim = 0)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return in_batch.to(device), nt_batch.to(device)