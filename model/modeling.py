"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
from builtins import NotImplementedError
import copy
import json
import logging
from io import open
from ntpath import join
import ipdb
from .transformer import GELU
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm 
from torch.nn.modules import dropout

import math
from .transformer import TransformerEncoder, clones
import random
import numpy as np

import time
from utils.logger import LOGGER
import torch.nn.functional as F
from time import time




class VALORConfig(object):
    def __init__(self,
                 config):
        
        if isinstance(config, dict):
            for key, value in config.items():
                self.__dict__[key] = value

        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `VALORConfig` from a
           Python dictionary of parameters."""
        config = VALORConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `VALORConfig` from a json file of parameters."""
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


class VALORPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self,  *inputs, **kwargs):
        super().__init__()
        # if not isinstance(config, VALORConfig):
        #     raise ValueError(
        #         "Parameter config in `{}(config)` should be an instance of "
        #         "class `VALORConfig`. To create a model from a Google "
        #         "pretrained model use "
        #         "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
        #             self.__class__.__name__, self.__class__.__name__
        #         ))
        # self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=0.02)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, opts, state_dict, *inputs, **kwargs):
        model = cls(opts, *inputs, **kwargs)
        missing_keys,unexpected_keys = model.load_state_dict(state_dict,strict=False)
        if state_dict != {}:
            # print(state_dict)
            LOGGER.info(f"Unexpected keys {unexpected_keys}")
            LOGGER.info(f"missing_keys  {missing_keys}")
        return model


def valid(x):
    return x is not None


class TokenMasker(nn.Module):
    def __init__(self, mask_token = -1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start,range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    
    def perform_mask(self, tokens, mask_prob):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1
        
        


        labels = -np.ones(tokens.shape, dtype=np.int64)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))   
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        # labels = -np.ones(tokens.shape, dtype=np.int64)
        # for i in range(tokens.shape[0]):
        #     for j in range(tokens.shape[1]):                
        #         if mask_indicator[i][j] == 1 :
        #             labels[i][j] = tokens[i][j]
        #             tokens[i][j] = self.mask_token  ### e-6 have no idea why too much                         
                    

        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels



class TokenMaskerWoReplace(nn.Module):
    def __init__(self, mask_token = -1):
        super().__init__()
        self.mask_token = mask_token
 

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    
    def perform_mask(self, tokens, mask_prob):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1
        
        


        labels = -np.ones(tokens.shape, dtype=np.int64)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.9:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        # labels = -np.ones(tokens.shape, dtype=np.int64)
        # for i in range(tokens.shape[0]):
        #     for j in range(tokens.shape[1]):                
        #         if mask_indicator[i][j] == 1 :
        #             labels[i][j] = tokens[i][j]
        #             tokens[i][j] = self.mask_token  ### e-6 have no idea why too much                         
                    

        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels



class BERTPredictionHead(nn.Module):
    def __init__(self, embedding_weights):
        super().__init__()
        self.hidden_size = embedding_weights.size(1)
        self.vocab_size = embedding_weights.size(0)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = GELU()
        self.layernorm = FusedLayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder.weight = embedding_weights
        


    def forward(self, sequence_output):
        # import ipdb 
        # ipdb.set_trace()
    
        sequence_output = self.dense(sequence_output)

        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 
        return prediction_scores






class Config():
    def __init__(self):
        self.void = 'void'






base_cfg = Config()
base_cfg.attention_dropout = 0.1
base_cfg.hidden_act= "gelu"
base_cfg.hidden_dropout= 0.1
base_cfg.hidden_size= 768
base_cfg.initializer_range = 0.02
base_cfg.intermediate_size = 3072
base_cfg.num_attention_heads = 12
base_cfg.num_hidden_layers = 12


class VALORModel(VALORPreTrainedModel):

    def __init__(self, opts):
        super().__init__()
        config = opts
        self.config = config

        ### txt embedding and txt encoder
        self.video_encoder_type = config.video_encoder_type
        self.txt_encoder_type = config.txt_encoder_type
        self.audio_encoder_type = config.audio_encoder_type
        self.multimodal_encoder_type = config.multimodal_encoder_type
        self.multimodal_use_cross_attn = getattr(config,'multimodal_use_cross_attn',False)
        self.initial_multimodal = config.initial_multimodal
        self.initial_vision = config.initial_vision


        #### construct clip
        clip_type=None
        for i in (config.txt_encoder_type, config.video_encoder_type):
            if i.startswith('clip'): 
                clip_type = i
        if clip_type is not None:  
            self.load_clip_model( clip_type, opts)


        ##### construct vision encoder 
        if self.video_encoder_type.startswith('videoswin'):
            self.load_videoswin_model(config)

            
        elif  self.video_encoder_type.startswith('clip'):
            if self.video_encoder_type.startswith('clip_vit_base'):
                self.video_dim = 768
            elif self.video_encoder_type.startswith('clip_vit_large'):
                self.video_dim = 1024
            elif self.video_encoder_type.startswith('clip_vit_huge'):
                self.video_dim = 1280
            if config.frozen_vision:
                for k,v in self.clip_model.named_parameters():
                    if 'visual' in k:
                        v.requires_grad = False 
        else:
            raise NotImplementedError

        ### construct  audio encoder
        if self.audio_encoder_type.startswith('ast'):
            self.load_ast_model(base_cfg,config)
        else:
            raise NotImplementedError
        ### construct multimodal encoder
        if self.multimodal_encoder_type.startswith('bert'):
            self.load_bert_model(config)
        else:
            raise NotImplementedError

        ### construct text encoder

        self.construct_text_model(config)
        
        self.video_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.video_frame_embedding = nn.Parameter(0.02 * torch.randn(1, 32, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, 32, self.multimodal_dim))  
        #self.video_ln = FusedLayerNorm(self.multimodal_dim, eps=1e-12)
        self.hidden_trans_video_multimodal = None
        self.hidden_trans_audio_multimodal = None
        if self.video_dim != self.multimodal_dim:
            self.hidden_trans_video_multimodal = nn.Sequential(nn.Linear(self.video_dim, self.multimodal_dim),FusedLayerNorm(self.multimodal_dim, eps=1e-12))
        if self.audio_dim != self.multimodal_dim:
            self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),FusedLayerNorm(self.multimodal_dim, eps=1e-12))


        
    def get_task_prompt(self, sentence, batch_size=0,cls_prompt=False, type=None):

        if type is None:
            type = self.multimodal_encoder_type
        if  not type.startswith('clip'):
            sentence = self.tokenizer.tokenize(sentence)
            sentence = self.tokenizer.convert_tokens_to_ids(sentence)
            task_prompt = [self.bos_token] + sentence + [self.eos_token]
        
        else:
            sentence = self.clip_tokenizer.encode(sentence)
            task_prompt = [self.sot_token] + sentence + [self.eot_token]
        if not cls_prompt:
            task_prompt = torch.tensor(task_prompt).unsqueeze(0).expand(batch_size,-1).long().cuda()
        return task_prompt
      


    def pool_text_for_contra(self, feature, txt_tokens=None, contra_type = None):
        contra_type = contra_type if contra_type is not None else self.contra_type
        if contra_type == 'coarse':
            if self.txt_encoder_type.startswith('bert'):
                return feature[:,0]
            elif self.txt_encoder_type.startswith('clip'):
                feature = feature[torch.arange(txt_tokens.shape[0]), txt_tokens.argmax(dim=-1)]
            return feature 
        else:
            return feature

    def pool_video_for_contra(self, feature, contra_type = None):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.video_encoder_type.startswith('clip') or self.video_encoder_type.startswith('beit') or self.video_encoder_type.startswith('vit'):
            feature = feature[:,:,0]
        elif self.video_encoder_type.startswith('videoswin'):
            feature = feature.mean(dim=2)
        contra_type = contra_type if contra_type is not None else self.contra_type

        if contra_type == 'coarse':
            return torch.mean(feature, dim=1)
        else:
            return feature  

    def pool_audio_for_contra(self, feature, contra_type = None):
        if self.audio_encoder_type.startswith('ast'):
            feature = feature[:,:,0]
        else:
            raise NotImplementedError
        contra_type = contra_type if contra_type is not None else self.contra_type
        if contra_type == 'coarse':
            return torch.mean(feature, dim=1)
        else:
            return feature  


    def get_text_tokens(self,txt_tokens, type):
        if txt_tokens == None :
            return txt_tokens
        if type.startswith('clip'):
            return txt_tokens['clip_tokens']
        elif type.startswith('bert'):
            return txt_tokens['bert_tokens']

   
    def contrastive_loss(self, score_matrix):  ### labels for unicl 
        
        if self.video_encoder_type.startswith('clip'):
            temp = 1./ self.clip_model.logit_scale.exp()
        else:
            temp = self.contra_temp

     
        score_matrix = score_matrix / temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)      
        loss1 = matrix1.diag()
        loss2 = matrix2.diag()
        contra_loss = torch.mean(torch.cat((loss1,loss2), dim=0))

        return contra_loss



    def forward_txt_encoder(self, txt_tokens,task_prompt=None):
        
        if self.txt_encoder_type.startswith('bert'):
            txt_output = self.txt_encoder(txt_tokens,task_prompt,casual = False)
        elif self.txt_encoder_type.startswith('clip'):
            txt_output = self.clip_model.encode_text(txt_tokens, casual = True)
        else:
            raise NotImplementedError()

        return txt_output


    def forward_video_encoder(self, video_pixels):   ### b,n,3,h,w

        b,n,_,h,w = video_pixels.shape
        if self.video_encoder_type.startswith('videoswin'):
            video_output = self.video_encoder(video_pixels.transpose(1,2))
            video_output = video_output.permute(0, 2, 3, 4, 1)
            video_output = video_output.reshape(video_output.shape[0], video_output.shape[1], -1,video_output.shape[-1])

        elif self.video_encoder_type.startswith('clip'):
            
            video_output = self.clip_model.encode_image(video_pixels.reshape(b*n,3,h,w))
            video_output = video_output.reshape(b,-1,*video_output.shape[-2:])
            
        else:
            raise NotImplementedError()

        return video_output  #### B , n , x ,C  n = self.frame_num

        
    def forward_audio_encoder(self, audio_spectrograms):

        if self.audio_encoder_type.startswith('ast'): 
            b,n,h,w, = audio_spectrograms.shape
            audio_spectrograms = audio_spectrograms.reshape(-1,*audio_spectrograms.shape[2:])
            audio_embeddings = self.audio_embeddings(audio_spectrograms) 
            audio_output,_ = self.audio_encoder(audio_embeddings)
            audio_output = audio_output.reshape(b,n,-1,audio_output.shape[-1])
            
        else:
            raise NotImplementedError()
              
        return audio_output




    def get_multimodal_forward_input_video(self, video_output):
        b,n,x,c = video_output.shape
        if self.hidden_trans_video_multimodal is not None:
            video_output = self.hidden_trans_video_multimodal(video_output)  
        video_output =  video_output + self.video_frame_embedding[:,:video_output.shape[1],:].unsqueeze(-2)
        video_output = video_output.reshape(b,-1,self.multimodal_dim) 
        video_output =  video_output + self.video_type_embeddings

        return video_output

    def get_multimodal_forward_input_audio(self, audio_output):
        b,n,x,c = audio_output.shape
        if self.hidden_trans_audio_multimodal is not None:
            audio_output = self.hidden_trans_audio_multimodal(audio_output)
        audio_output =  audio_output + self.audio_frame_embedding[:,:audio_output.shape[1],:].unsqueeze(-2)
        audio_output = audio_output.reshape(b,-1,self.multimodal_dim)
        audio_output = audio_output + self.audio_type_embeddings
        return audio_output


    def forward_multimodal_encoder(self, *args,**kwargs):

        if self.multimodal_encoder_type.startswith('clip'):
            return self.clip_model.encode_text(*args,**kwargs)
        else:
            return self.multimodal_encoder(*args,**kwargs)

    def initialize_audio_weights(self):
        # if self.audio_encoder_type == 'ast':
        ast_weight = torch.load('./pretrained_weights/audioset_10_10_0.4593.pth',map_location='cpu')
        audio_weight = {}
        audio_weight['audio_embeddings.cls_token']  = ast_weight['module.v.cls_token'] 
        audio_weight['audio_embeddings.distill_token']  = ast_weight['module.v.dist_token'] 
        audio_weight['audio_embeddings.first_conv.weight'] =  ast_weight['module.v.patch_embed.proj.weight']  ### need to permute?
        audio_weight['audio_embeddings.first_conv.bias'] = ast_weight['module.v.patch_embed.proj.bias']
        pos_weight = ast_weight['module.v.pos_embed'][0]
        pos_weight_cls = pos_weight[0:1]
        pos_weight_oth = pos_weight[2:]   #### give up the distilled token
        pos_weight_oth = pos_weight_oth.reshape(12, 101,-1).permute(2,0,1).unsqueeze(0)
        tar_patch_num_height = self.config.audio_melbins // self.config.audio_patch_size
        tar_patch_num_width = self.config.audio_target_length // self.config.audio_patch_size
        pos_weight_oth = F.interpolate(pos_weight_oth, size = (tar_patch_num_height,tar_patch_num_width),mode='bilinear').squeeze().permute(1,2,0).reshape(-1,768)
        pos_weight_oth = torch.cat((pos_weight_cls,pos_weight_oth),dim=0)
        audio_weight['audio_embeddings.position_embeddings.weight'] = pos_weight_oth

        for  i in range(12):
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][:768,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][:768]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][768:2*768,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][768:2*768]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][2*768:,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][2*768:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = ast_weight['module.v.blocks.'+str(i)+'.attn.proj.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.proj.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc1.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc1.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc2.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc2.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.weight']  = ast_weight['module.v.blocks.'+str(i)+'.norm1.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.bias']  = ast_weight['module.v.blocks.'+str(i)+'.norm1.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.weight']  = ast_weight['module.v.blocks.'+str(i)+'.norm2.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.bias'] = ast_weight['module.v.blocks.'+str(i)+'.norm2.bias']
        audio_weight['audio_encoder.last_layernorm.weight'] = ast_weight['module.v.norm.weight']
        audio_weight['audio_encoder.last_layernorm.bias'] = ast_weight['module.v.norm.bias']

        missing_keys, unexpected_keys = self.load_state_dict(audio_weight, strict=False)
        #LOGGER.info(f'missing_keys in audio encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in audio encoder: {unexpected_keys}')
        del(ast_weight)
        del(audio_weight)





    def load_clip_model(self, clip_type,opts):
        from .clip import build_model
        from .clip import Transformer
        if clip_type == 'clip_vit_base_16':
            clip_weight = torch.jit.load('./pretrained_weights/clip-vit-base-16.pt', map_location='cpu')
        elif clip_type == 'clip_vit_base_32':
            clip_weight = torch.jit.load('./pretrained_weights/clip-vit-base-32.pt', map_location='cpu')
        elif clip_type == 'clip_vit_large_14':
            clip_weight = torch.jit.load('./pretrained_weights/clip-vit-large-14.pt', map_location='cpu')
        elif clip_type == 'clip_vit_large_14_336px':
            clip_weight = torch.jit.load('./pretrained_weights/clip-vit-large-14-336px.pt', map_location='cpu')

        clip_weight = clip_weight.state_dict()
        self.clip_model = build_model(clip_weight, opts.video_resolution,opts.checkpointing).float()
        

    def load_videoswin_model(self,config):
        from .videoswin import SwinTransformer3D
        assert self.video_encoder_type in ['videoswin_small_k400_1k','videoswin_base_k400_22k',
                                            'videoswin_base_k600_22k','videoswin_base_k400_1k']
        self.time_stride = 1

        if self.video_encoder_type.startswith('videoswin_small'):
            self.video_encoder = SwinTransformer3D(time_stride = self.time_stride, embed_dim = 96, num_heads=[3, 6, 12, 24],checkpointing=config.checkpointing)
            self.video_dim = 768
        elif self.video_encoder_type.startswith('videoswin_base'):
            self.video_encoder = SwinTransformer3D(time_stride = self.time_stride, embed_dim=128, num_heads=[4, 8, 16, 32],checkpointing=config.checkpointing)
            self.video_dim = 1024
            #self.hidden_size_adjust = nn.Sequential(nn.Linear(1024,768),FusedLayerNorm(768, eps=1e-12))


        if self.video_encoder_type == 'videoswin_small_k400_1k':
            videoswin_weight = torch.load('./pretrained_weights/ckpt_video-swin.pt',map_location='cpu')
        elif self.video_encoder_type == 'videoswin_base_k400_1k':
            videoswin_weight = torch.load('./pretrained_weights/videoswin_base_k400_1k.pth',map_location='cpu')
        elif self.video_encoder_type == 'videoswin_base_k400_22k':
            videoswin_weight = torch.load('./pretrained_weights/videoswin_base_k400_22k.pth',map_location='cpu')
        elif self.video_encoder_type == 'videoswin_base_k600_22k':
            videoswin_weight = torch.load('./pretrained_weights/videoswin_base_k600_22k.pth',map_location='cpu')
        missing_keys, unexpected_keys = self.video_encoder.load_state_dict(videoswin_weight)
        del(videoswin_weight)
        #LOGGER.info(f'missing_keys in video encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in video encoder: {unexpected_keys}')


    def load_ast_model(self,base_cfg,config):
        model_cfg_audio = copy.deepcopy(base_cfg)
        model_cfg_audio.checkpointing = config.checkpointing
        self.audio_embeddings = AudioEmbeddings(model_cfg_audio, config)
        self.audio_encoder = TransformerEncoder(model_cfg_audio, mode='prenorm')
        self.initialize_audio_weights()
        self.audio_dim = 768

    def load_bert_model(self,config):
        
        from model.bert import BertModel, BertConfig
        bert_base_config = BertConfig.from_json_file("./pretrained_weights/bert_base_uncased_config.json")
        bert_large_config = BertConfig.from_json_file("./pretrained_weights/bert_large_uncased_config.json")
        if self.multimodal_encoder_type == 'bert_base_uncased':
            #bertconfig = BertConfig.from_json_file("./pretrained_weights/bert_base_uncased_config.json")
            bertconfig = bert_base_config
            bert_weight = torch.load('./pretrained_weights/bert-base-uncased.bin',map_location='cpu')
            self.multimodal_dim = 768
        elif self.multimodal_encoder_type == 'bert_base_chinese':
            bertconfig = BertConfig.from_json_file("./pretrained_weights/bert_base_chinese_config.json")
            bert_weight = torch.load('./pretrained_weights/bert-base-chinese.bin',map_location='cpu')
            self.multimodal_dim = 768
        elif self.multimodal_encoder_type == 'bert_large_uncased':
            #bertconfig = BertConfig.from_json_file("./pretrained_weights/bert_large_uncased_config.json")
            bertconfig = bert_large_config
            bert_weight = torch.load('./pretrained_weights/bert-large-uncased.bin',map_location='cpu')
            self.multimodal_dim = 1024
                 
        bertconfig.checkpointing = config.checkpointing

        if self.multimodal_use_cross_attn:
            bertconfig.has_cross_attn = True
            bertconfig.cross_attn_type = config.cross_attn_type
    
        else:
            bertconfig.has_cross_attn = False
            bertconfig.cross_attn_type = None

        
        self.multimodal_encoder = BertModel(bertconfig)
        bert_weight = {k.replace('bert.','').replace('gamma','weight').replace('beta','bias') : v for k,v in bert_weight.items()}

        if self.initial_multimodal:
            missing_keys, unexpected_keys = self.multimodal_encoder.load_state_dict(bert_weight, strict=False)
            LOGGER.info(f'unexpected_keys in multimodal encoder: {unexpected_keys}')
            LOGGER.info(f'missing_keys in multimodal encoder: {missing_keys}')
        self.cls = BERTPredictionHead(self.multimodal_encoder.embeddings.word_embeddings.weight)
        cls_head_weight = {}
        cls_head_weight['dense.weight']  = bert_weight['cls.predictions.transform.dense.weight']
        cls_head_weight['dense.bias']  = bert_weight['cls.predictions.transform.dense.bias']
        cls_head_weight['layernorm.weight'] = bert_weight['cls.predictions.transform.LayerNorm.weight' ]
        cls_head_weight['layernorm.bias'] =bert_weight['cls.predictions.transform.LayerNorm.bias']
        cls_head_weight['decoder.weight'] = bert_weight['cls.predictions.decoder.weight']
        cls_head_weight['decoder.bias'] = bert_weight['cls.predictions.bias']
        if self.initial_multimodal:
            missing_keys, unexpected_keys = self.cls.load_state_dict(cls_head_weight)
            LOGGER.info(f'missing_keys in cls_head : {missing_keys}')
            LOGGER.info(f'unexpected_keys in cls_head : {unexpected_keys}')
            del(bert_weight)
            del(cls_head_weight)

        
        
        from model.bert_tokenizer import BertTokenizer
        if self.multimodal_encoder_type == 'bert_base_chinese':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        else:
            self.tokenizer = BertTokenizer("./pretrained_weights/bert-base-uncased-vocab.txt")
            
        self.tokenizer_type = 'bert'
        self.bos_token = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.eos_token = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.text_mask_token = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        self.text_masker = TokenMasker(mask_token = self.text_mask_token, range_start=106, range_end = 30522)

        if config.frozen_multimodal:
            for k,v in self.multimodal_encoder.named_parameters():
                if 'encoder' in k and not 'cross' in k:
                    v.requires_grad = False 
                if 'embeddings' in k and any(j in k for j in ['embeddings.word_embeddings','embeddings.position_embeddings','embeddings.token_type_embeddings','embeddings.LayerNorm']):
                    v.requires_grad = False 
            for k,v in self.cls.named_parameters():
                v.requires_grad = False 


    def construct_text_model(self,config):


        if self.txt_encoder_type.startswith('bert'):
            if config.share_txt_and_multimodal:
                self.txt_encoder = self.multimodal_encoder
                self.txt_dim = self.multimodal_dim
            else:
                from model.bert import BertModel, BertConfig
                assert self.txt_encoder_type == 'bert_base_uncased'
                bert_base_config = BertConfig.from_json_file("./pretrained_weights/bert_base_uncased_config.json")
                bertconfig = bert_base_config
                bert_weight = torch.load('./pretrained_weights/bert-base-uncased.bin',map_location='cpu')
                self.txt_dim = 768      
                bertconfig.checkpointing = config.checkpointing
                bertconfig.has_cross_attn = False
                bertconfig.cross_attn_type = None
                self.txt_encoder = BertModel(bertconfig)
                bert_weight = {k.replace('bert.','').replace('gamma','weight').replace('beta','bias') : v for k,v in bert_weight.items()}
                missing_keys, unexpected_keys = self.txt_encoder.load_state_dict(bert_weight, strict=False)
                LOGGER.info(f'unexpected_keys in multimodal encoder: {unexpected_keys}')
                LOGGER.info(f'missing_keys in multimodal encoder: {missing_keys}')



        elif self.txt_encoder_type.startswith('clip'):
            if self.txt_encoder_type.startswith('clip') or self.multimodal_encoder_type.startswith('clip'):
                from .clip_tokenizer import SimpleTokenizer
                self.clip_tokenizer = SimpleTokenizer()
                self.sot_token = self.clip_tokenizer.encoder["<|startoftext|>"]
                self.eot_token = self.clip_tokenizer.encoder["<|endoftext|>"]

            if self.txt_encoder_type.startswith('clip_vit_base'):
                self.txt_dim = 512
            elif self.txt_encoder_type.startswith('clip_vit_large'):
                self.txt_dim = 768
            elif self.txt_encoder_type.startswith('clip_vit_huge'):
                self.txt_dim = 1024
            
        else:
            raise NotImplementedError












class AudioEmbeddings(nn.Module):
    def __init__(self, model_cfg_audio, config):
        super().__init__()
        
        self.patch_size = config.audio_patch_size
        self.token_length_per_frame = (config.audio_melbins // self.patch_size) * (config.audio_target_length // self.patch_size)
        self.first_conv = nn.Conv2d(1, model_cfg_audio.hidden_size, kernel_size = self.patch_size, 
                                    stride = self.patch_size, padding=0)
        self.position_embeddings = nn.Embedding(self.token_length_per_frame + 1, model_cfg_audio.hidden_size)
        self.dropout = nn.Dropout(model_cfg_audio.hidden_dropout)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, model_cfg_audio.hidden_size))

    def forward(self, audio_spectrograms):  ### shape Bxn_sample_128x512
        
        audio_spectrograms = self.first_conv(audio_spectrograms.unsqueeze(1))
        b,c,_,_=audio_spectrograms.shape
        audio_tokens = audio_spectrograms.permute(0,2,3,1).reshape(b,-1,c)        
        cls_token = self.cls_token.expand(b,-1,-1)
        audio_tokens = torch.cat((cls_token,audio_tokens),dim=1)
        audio_pos_ids = list(range(self.token_length_per_frame + 1))
        audio_pos_ids = torch.tensor(audio_pos_ids, dtype=torch.long, device=audio_spectrograms.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(audio_pos_ids)
        embeddings = audio_tokens + position_embeddings 
        embeddings = self.dropout(embeddings)
        return embeddings

        
def trans(x):
    return torch.from_numpy(x)


