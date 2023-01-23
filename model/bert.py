# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils import text_file

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.prompt_embedding = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        #### for visual 
       
        
    def forward(self, input_ids, token_type = None, full_masker=False):
        # import ipdb 
        # ipdb.set_trace()
        if token_type  == 'prompt' or token_type is None:
            seq_length = input_ids.size(1)
            
            if token_type is None:
                if not full_masker:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                else:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                    position_ids[seq_length//2:] = position_ids[:seq_length//2] + 1

                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeddings = self.token_type_embeddings(token_type_ids)
            elif token_type == 'prompt':
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeddings = self.prompt_embedding(token_type_ids)       
            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            

            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        
       
        
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, use_cache=False, cache=None, cache_first=False, layer_num=0, cache_type='unimlm'):
        mixed_query_layer = self.query(hidden_states)   ### b,n,c
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # b,h,n,c
        key_layer = self.transpose_for_scores(mixed_key_layer)    # b,h,n,c
        value_layer = self.transpose_for_scores(mixed_value_layer) # b,h,n,c

        # if layer_num==0:
        #     import ipdb 
        #     ipdb.set_trace()


        if use_cache:

            if not cache_first:
                key_layer = torch.cat((key_layer,cache['key'][str(layer_num)]),dim=2)
                value_layer = torch.cat((value_layer,cache['value'][str(layer_num)]),dim=2)
            if cache_type == 'unimlm':
                idx = 2
            elif cache_type == 'lm':
                idx = 1
            cache['key'][str(layer_num)] = torch.cat((key_layer[:,:,0:1],key_layer[:,:,idx:]),dim=2)  ### b,h,n,c
            cache['value'][str(layer_num)] = torch.cat((value_layer[:,:,0:1],value_layer[:,:,idx:]),dim=2)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) ## b,h,n,n
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)   ##b,h,n,c
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  ###b,n,h,c
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, cache
      

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(cross_hidden_states)
        mixed_value_layer = self.value(cross_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossOutput(nn.Module):
    def __init__(self, config):
        super(BertCrossOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.gating = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ### do not need any gates, seriously bad effect s
        #hidden_states = self.LayerNorm(self.gating * hidden_states + input_tensor)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, attn_type):
        super(BertAttention, self).__init__()
        self.attn_type = attn_type 
        if self.attn_type == 'self':
            self.self = BertSelfAttention(config)
            self.output = BertSelfOutput(config)
        elif self.attn_type == 'cross':
            self.cross = BertCrossAttention(config)
            self.output = BertCrossOutput(config)
        
        

    def forward(self, input_tensor, attention_mask, corss_hidden_states, use_cache=False, cache=None, cache_first=False, layer_num=0,cache_type='unimlm'):
        if self.attn_type == 'self':
            self_output, cache = self.self(input_tensor, attention_mask, use_cache=use_cache, cache=cache, cache_first=cache_first,layer_num=layer_num,cache_type=cache_type)
            attention_output = self.output(self_output, input_tensor)
        elif self.attn_type == 'cross':
            cross_output = self.cross(input_tensor, corss_hidden_states)
            attention_output = self.output(cross_output, input_tensor)
        return attention_output, cache


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, attn_type='self')
        self.has_cross_attn = config.has_cross_attn
        self.cross_attn_type = config.cross_attn_type
        if self.has_cross_attn:
            assert self.cross_attn_type in ['video_audio','audio_video','va_parallel','va_concate']
            if self.cross_attn_type == 'va_concate':
                self.cross_attn = BertAttention(config, attn_type='cross')
            else:
                self.cross_attn_v = BertAttention(config, attn_type='cross')
                self.cross_attn_a = BertAttention(config, attn_type='cross')
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        

    def forward(self, hidden_states, attention_mask, video_feat, audio_feat, use_cache=False, cache=None,
                                        cache_first=False, layer_num=0, cache_type='unimlm'):

        attention_output, cache = self.attention(hidden_states, attention_mask, None, use_cache=use_cache, cache=cache,
                                        cache_first=cache_first, layer_num=layer_num,cache_type=cache_type)
        # if self.has_cross_attn and cross_hidden_states is not None:
        #     attention_output, cache = self.cross_attn(attention_output, None, cross_hidden_states) 
        if self.has_cross_attn:
            if self.cross_attn_type == 'va_concate':
                if video_feat is not None and audio_feat is not None:
                    attention_output, cache = self.cross_attn(attention_output, None, torch.cat((video_feat,audio_feat),dim=1))
                
                elif video_feat is None and audio_feat is not None:
                    attention_output, cache = self.cross_attn(attention_output, None, audio_feat)
                elif video_feat is not  None and audio_feat is  None:
                    attention_output, cache = self.cross_attn(attention_output, None, video_feat)
                elif video_feat is None and audio_feat is  None:
                    pass

            elif self.cross_attn_type == 'va_parallel':
                if video_feat is not None and audio_feat is not None:
                    attention_output_v, cache = self.cross_attn_v(attention_output, None, video_feat)
                    attention_output_a, cache = self.cross_attn_a(attention_output, None, audio_feat)
                    attention_output = attention_output_v + attention_output_a
                elif video_feat is None and audio_feat is not None:
                    attention_output, cache = self.cross_attn_a(attention_output, None, audio_feat)
                elif video_feat is not  None and audio_feat is  None:
                    attention_output, cache = self.cross_attn_v(attention_output, None, video_feat)
                elif video_feat is None and audio_feat is  None:
                    pass

            elif self.cross_attn_type == 'video_audio':
                if video_feat is not None and audio_feat is not None:
                    attention_output, cache = self.cross_attn_v(attention_output, None, video_feat)
                    attention_output, cache = self.cross_attn_a(attention_output, None, audio_feat)
                    
                elif video_feat is None and audio_feat is not None:
                    attention_output, cache = self.cross_attn_a(attention_output, None, audio_feat)
                elif video_feat is not  None and audio_feat is  None:
                    attention_output, cache = self.cross_attn_v(attention_output, None, video_feat)
                elif video_feat is None and audio_feat is  None:
                    pass         


            elif self.cross_attn_type == 'audio_video':
                if video_feat is not None and audio_feat is not None:
                    attention_output, cache = self.cross_attn_a(attention_output, None, audio_feat)
                    attention_output, cache = self.cross_attn_v(attention_output, None, video_feat)
                elif video_feat is None and audio_feat is not None:
                    attention_output, cache = self.cross_attn_a(attention_output, None, audio_feat)
                elif video_feat is not  None and audio_feat is  None:
                    attention_output, cache = self.cross_attn_v(attention_output, None, video_feat)
                elif video_feat is None and audio_feat is  None:
                    pass         
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,cache



class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.checkpointing = config.checkpointing

    def forward(self, hidden_states, attention_mask, video_feat, audio_feat,use_cache=False,cache=None,
                                        cache_first=False,cache_type='unimlm'):
        for idx, layer_module in enumerate(self.layer):
            if self.checkpointing:
                # print(1)
                hidden_states,cache = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask, video_feat,audio_feat,use_cache,cache,
                                        cache_first, idx, cache_type)
            else:
                hidden_states,cache = layer_module(hidden_states, attention_mask,video_feat, audio_feat,use_cache,cache,
                                        cache_first, idx, cache_type)
            
        return hidden_states,cache


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.has_cross_attn = config.has_cross_attn
        self.apply(self.init_bert_weights)
        
        
    def forward(self, tokens, task_prompt = None, video_feat = None, audio_feat=None, \
                             casual=False, cache = None, use_cache = False, \
                                    cache_first=False, token_type=None, cache_type='unimlm', use_cross_attn=True,\
                                      full_masker=False):
        # import ipdb 
        # ipdb.set_trace()

        if (not self.has_cross_attn) or (self.has_cross_attn and not use_cross_attn):
            
            

                    
                    
                    
            if not use_cache or (use_cache and cache_first):
                if tokens is not None:
                    embedding_output = self.embeddings(tokens, token_type, full_masker)
                    input_feat = embedding_output
                    token_len = input_feat.shape[1]
                    attention_mask_token = (tokens != 0).long()
                    attention_mask = attention_mask_token
               
                if task_prompt is not None:
                    
                    if tokens is not None:
                        prompt_embedding_output = self.embeddings(task_prompt, 'prompt')
                        input_feat = torch.cat((input_feat, prompt_embedding_output),dim=1)
                        attention_mask_prompt = (task_prompt != 0).long()
                        attention_mask = torch.cat((attention_mask, attention_mask_prompt),dim=1)  
                    else:
                        prompt_embedding_output = self.embeddings(task_prompt, 'prompt')
                        input_feat = prompt_embedding_output
                        attention_mask_prompt = (task_prompt != 0).long()
                        attention_mask = attention_mask_prompt
                    
                if video_feat is not None:
                    input_feat = torch.cat((input_feat, video_feat),dim=1)
                    attention_mask_video = torch.ones(*video_feat.shape[:2]).to(attention_mask)
                    attention_mask = torch.cat((attention_mask, attention_mask_video),dim=1)
                if audio_feat is not None:
                    input_feat = torch.cat((input_feat, audio_feat),dim=1)
                    attention_mask_audio = torch.ones(*audio_feat.shape[:2]).to(attention_mask)
                    attention_mask = torch.cat((attention_mask, attention_mask_audio),dim=1)
               

                #attn_mask = attention_mask.clone()
                total_len = attention_mask.shape[1]
                attention_mask = attention_mask.unsqueeze(1).expand(-1, total_len, -1).clone()
                if casual and (tokens is not None):
                    attention_mask[:, : token_len, : token_len] = torch.tril(attention_mask[:, : token_len, : token_len])
                    attention_mask[:, token_len:, : token_len] = 0
                attention_mask = attention_mask.unsqueeze(1)
                attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -10000.0
                    
            

                sequence_output, cache = model(input_feat,
                                            attention_mask,
                                            video_feat=None,
                                            audio_feat=None,
                                            use_cache=use_cache,
                                            cache=cache,
                                            cache_first=cache_first,
                                            cache_type=cache_type)

                if not use_cache:
                    return sequence_output
                elif cache_type=='unimlm':
                    cache['attn_masks'] = attention_mask[:,:,:2,:]
                    return sequence_output, cache
                elif cache_type=='lm':
                    cache['attn_masks'] = attention_mask[:,:,:1,:]
                    return sequence_output, cache

            else:  #### use cache and not cache first
                ### use_cache : only for unimlm 
                if cache_type=='unimlm':
                    padding_one = torch.zeros(*cache['attn_masks'].shape[:3],1).to(cache['attn_masks'])
                    cache['attn_masks'] = torch.cat((cache['attn_masks'][:,:,:,:2],padding_one,cache['attn_masks'][:,:,:,2:]),dim=-1)
                    embedding_output = self.embeddings(tokens, token_type)
                    input_feat = embedding_output[:,-2:]
                elif cache_type=='lm':
                    padding_one = torch.zeros(*cache['attn_masks'].shape[:3],1).to(cache['attn_masks'])
                    cache['attn_masks'] = torch.cat((cache['attn_masks'][:,:,:,:1],padding_one,cache['attn_masks'][:,:,:,1:]),dim=-1)
                    embedding_output = self.embeddings(tokens, token_type)
                    input_feat = embedding_output[:,-1:]
                sequence_output, cache = model(input_feat,
                                            cache['attn_masks'],
                                            video_feat=None,
                                            audio_feat=None,
                                            use_cache=True,
                                            cache=cache,
                                            cache_first=False,
                                            cache_type=cache_type)
                
                return sequence_output, cache

        else:
            assert use_cache == False
            # print(1)
            embedding_output = self.embeddings(tokens, token_type,full_masker)
            input_feat = embedding_output
            token_len = input_feat.shape[1]
            attention_mask_token = (tokens != 0).long()
            attention_mask = attention_mask_token
            # if not self.has_cross_attn:

            if task_prompt is not None:
                
                prompt_embedding_output = self.embeddings(task_prompt, 'prompt')
                input_feat = torch.cat((input_feat, prompt_embedding_output),dim=1)
                attention_mask_prompt = (task_prompt != 0).long()
                attention_mask = torch.cat((attention_mask, attention_mask_prompt),dim=1)            





            total_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, total_len, -1).clone()
            if casual:
                if full_masker:
   
                    attention_mask[:, : token_len//2, : token_len//2] = torch.tril(attention_mask[:, : token_len//2, : token_len//2])
                    attention_mask[:, : token_len//2, token_len//2:token_len] = 0
                    attention_mask[:, token_len//2:token_len, :token_len//2] = torch.tril(attention_mask[:, token_len//2:token_len, :token_len//2])
                    attention_mask[:, token_len//2:token_len, token_len//2:token_len] = torch.eye(token_len//2)
                    attention_mask[:, token_len:, : token_len] = 0
                    
                else:
                    attention_mask[:, : token_len, : token_len] = torch.tril(attention_mask[:, : token_len, : token_len])
                    attention_mask[:, token_len:, : token_len] = 0
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            

            sequence_output, cache = self.encoder(input_feat,
                                        attention_mask,
                                        video_feat=video_feat,
                                        audio_feat=audio_feat,
                                        use_cache=use_cache,
                                        cache=cache,
                                        cache_first=cache_first)

            return sequence_output
