"""
BERT layers from the huggingface implementation
(https://github.com/huggingface/transformers)
"""
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import copy
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import torch.nn.functional as F
import ipdb

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output




class TransformerLayer(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ff_layer = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layernorm1 = LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(config.hidden_size, eps=1e-12)
        self.mode = mode

    def forward(self, hidden_states, attention_mask):
        if self.mode == 'prenorm':
            return self.forward_prenorm(hidden_states, attention_mask)
        elif self.mode == 'postnorm':
            return self.forward_postnorm(hidden_states, attention_mask)
        else:
            raise NotImplementedError
    
    def forward_prenorm(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        attention_output = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attention_output)

        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        ff_output = self.ff_layer(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states

    def forward_postnorm(self, hidden_states, attention_mask):
        residual = hidden_states
        attention_output = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attention_output)
        hidden_states = self.layernorm1(hidden_states)

        residual = hidden_states
        ff_output = self.ff_layer(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        hidden_states = self.layernorm2(hidden_states)

        return hidden_states


def clones(x,times):
    return nn.ModuleList([copy.deepcopy(x) for i in range(times)])



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linears = clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.head_num = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout=nn.Dropout(config.attention_dropout)


    def forward(self,q,k,v,mask=None):
        batch_size=q.shape[0]
        q,k,v=[layer(x).view(batch_size,-1,self.head_num, self.hidden_size//self.head_num).transpose(1,2) \
                 for layer,x in zip(self.linears,(q,k,v))]
        norm_d=q.shape[-1]
        att_map=torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(norm_d)
        if mask is not None:
            att_map=att_map + mask        
        att_map=F.softmax(att_map,dim=-1)
        # import ipdb
        # if att_map.shape[-1] == 45:
        #     ipdb.set_trace()

        att_map=self.dropout(att_map)
        attn_output = self.linears[-1](torch.matmul(att_map,v).transpose(1,2).contiguous().view(batch_size,-1,self.hidden_size))
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1=nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2=nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = GELU()


    def forward(self,x):
        return self.linear2((self.activation(self.linear1(x))))



class TransformerEncoder(nn.Module):
    def __init__(self, config, mode = 'prenorm'):
        super().__init__()
        layer = TransformerLayer(config, mode)
        self.mode = mode
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])
        if self.mode == 'prenorm':
            self.last_layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.checkpointing = config.checkpointing
    def forward(self, input_, attention_mask=None, cross_hidden_states=None,
                                            use_cache=False,
                                            cache=None,
                                            cache_first=False,
                                            cache_type=None):
        hidden_states = input_
        for layer_module in self.layer:
            if self.checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)
            else:
                hidden_states = layer_module(hidden_states, attention_mask)

        if self.mode == 'prenorm':
            hidden_states = self.last_layernorm(hidden_states)
        return hidden_states, cache

