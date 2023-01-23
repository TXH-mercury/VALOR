"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

distributed API using Horovod
Modified from OpenNMT's native pytorch distributed utils
(https://github.com/OpenNMT/OpenNMT-py)
"""
import math
import pickle

import torch
import torch.distributed as dist
from time import time
from torch.autograd import Function 
from torch.utils.data.distributed import DistributedSampler

# class ddp_allgather_with_grads(Function):
#     @staticmethod
#     def forward(ctx, x):
#         x = x.cuda()
#         x_list = [torch.zeros_like(x) for i in range(dist.get_world_size())]
#         dist.all_gather(x_list, x)
#         x = torch.cat(x_list,dim=0) 
#         x.requires_grad = True 
#         return x 

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_x = None 
        
#         if grad_output is not None:
#             grad_output.detach()
#             grad_x = grad_output.chunk(dist.get_world_size(),dim=0)[dist.get_rank()]
        
#         return grad_x

class ddp_allgather_with_grads(Function):
    @staticmethod
    def forward(ctx, x):
        tmp_input = x.cuda()
        size = torch.tensor(tmp_input.shape[0]).cuda()
        size_list = [torch.zeros_like(size) for i in range(dist.get_world_size())]
        dist.all_gather(size_list, size)
        max_size = max(size_list).item()
        padding_size = max_size - size 
        if padding_size > 0 :
            padding_tensor = torch.zeros(padding_size,*tmp_input.shape[1:]).to(tmp_input)
            tmp_input = torch.cat((tmp_input, padding_tensor), dim = 0)
        tmp_list = [torch.zeros_like(tmp_input) for i in range(dist.get_world_size())]
        dist.all_gather(tmp_list, tmp_input)
        ctx.size = size_list
        output = []
        for t, s in zip(tmp_list, size_list):
            output.append(t[:s])
        output = torch.cat(output,dim=0) 
        output.requires_grad = True 
        return output 
        
         

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None 
        
        if grad_output is not None:
            grad_output.detach()
            #grad_x = grad_output.chunk(dist.get_world_size(),dim=0)[dist.get_rank()]
            start = sum(ctx.size[:dist.get_rank()])
            end = start + ctx.size[dist.get_rank()]
            grad_x = grad_output[start:end]
        return grad_x



######  with different batch_size ~
def ddp_allgather(input):
    tmp_input = input.cuda()
    size = torch.tensor(tmp_input.shape[0]).cuda()
    size_list = [torch.zeros_like(size) for i in range(dist.get_world_size())]
    dist.all_gather(size_list, size)
    max_size = max(size_list).item()
    padding_size = max_size - size 
    if padding_size > 0 :
        padding_tensor = torch.zeros(padding_size,*tmp_input.shape[1:]).to(tmp_input)
        tmp_input = torch.cat((tmp_input, padding_tensor), dim = 0)
    tmp_list = [torch.zeros_like(tmp_input) for i in range(dist.get_world_size())]
    dist.all_gather(tmp_list, tmp_input)
    output = []
    for t, s in zip(tmp_list, size_list):
        output.append(t[:s])
    output = torch.cat(output,dim=0) 
    return output 





def _encode(enc, max_size, use_max_size=False):
    enc_size = len(enc)
    enc_byte = max(math.floor(math.log(max_size, 256)+1), 1)
    if use_max_size:
        # this is used for broadcasting
        buffer_ = torch.cuda.ByteTensor(max_size+enc_byte)
    else:
        buffer_ = torch.cuda.ByteTensor(enc_size+enc_byte)
    remainder = enc_size
    for i in range(enc_byte):
        base = 256 ** (enc_byte-i-1)
        buffer_[i] = remainder // base
        remainder %= base
    buffer_[enc_byte:enc_byte+enc_size] = torch.ByteTensor(list(enc))
    return buffer_, enc_byte


def _decode(buffer_, enc_byte):
    size = sum(256 ** (enc_byte-i-1) * buffer_[i].item()
               for i in range(enc_byte))
    bytes_list = bytes(buffer_[enc_byte:enc_byte+size].tolist())
    shift = size + enc_byte
    return bytes_list, shift


_BUFFER_SIZE = 4096


def all_gather_list(data):
    """Gathers arbitrary data from all nodes into a list."""
    enc = pickle.dumps(data)

    enc_size = len(enc)
    max_size = ddp_allgather(torch.tensor([enc_size]).cuda()).max().item()
    in_buffer, enc_byte = _encode(enc, max_size)

    out_buffer = ddp_allgather(in_buffer[:enc_byte+enc_size])

    results = []
    for _ in range(dist.get_world_size()):
        bytes_list, shift = _decode(out_buffer, enc_byte)
        out_buffer = out_buffer[shift:]
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def any_broadcast(data, root_rank):
    """broadcast arbitrary data from root_rank to all nodes."""
    enc = pickle.dumps(data)

    max_size = ddp_allgather(torch.tensor([len(enc)]).cuda()).max().item()
    buffer_, enc_byte = _encode(enc, max_size, use_max_size=True)

    dist.broadcast(buffer_, root_rank)

    bytes_list, _ = _decode(buffer_, enc_byte)
    result = pickle.loads(bytes_list)
    return result



class DistributedSampler_wopadding(DistributedSampler):

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # if not self.drop_last:
        #     # add extra samples to make it evenly divisible
        #     padding_size = self.total_size - len(indices)
        #     if padding_size <= len(indices):
        #         indices += indices[:padding_size]
        #     else:
        #         indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        # else:
            # remove tail of data to make it evenly divisible.
        if self.drop_last:
            indices = indices[:self.total_size]
        #assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        # assert len(indices) == self.num_samples

        return iter(indices)
