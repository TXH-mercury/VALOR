"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
from math import ceil
import ipdb
import math 





def warmup_cosine(x, warmup_ratio):
    if x < warmup_ratio:
        return x/warmup_ratio
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup_ratio):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup_ratio:
        return x/warmup_ratio
    return 1.0

def warmup_linear(x, warmup_ratio):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup_ratio:
        return x/warmup_ratio
    return max((x-1.)/(warmup_ratio-1.), 0)

scheduler_dict = {'warmup_linear' : warmup_linear,
             'warmup_cosine' : warmup_cosine}

def get_lr_sched(global_step, opts):
    warmup_ratio = opts.warmup_ratio 
    current_ratio = global_step / opts.num_train_steps
    lr_ratio = scheduler_dict[opts.scheduler](current_ratio, warmup_ratio)
    return lr_ratio


