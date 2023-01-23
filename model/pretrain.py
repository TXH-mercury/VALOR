from collections import defaultdict
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from transformers import CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
import json
import torch.distributed as dist
from .modeling import VALORModel, VALORPreTrainedModel
import ipdb
import numpy as np
import random
from .transformer import MultiHeadAttention
from utils.logger import LOGGER
import yaml
from torchvision.transforms import *
import math
from time import time
from tqdm import tqdm
from utils.misc import NoOp
import os
from utils.distributed import any_broadcast
from utils.distributed import all_gather_list
from utils.distributed import ddp_allgather_with_grads, ddp_allgather
from apex.normalization.fused_layer_norm import FusedLayerNorm 
from scorer.scorer import Scorer

##ss
##ss
from .transformer import GELU


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)
class Config():
    def __init__(self):
        self.void = 'void'




class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None
        
    def forward(self, input_logits, target):
        input_logits = F.log_softmax(input_logits,dim=-1)
        size = input_logits.size(1)
        true_dist = input_logits.data.clone()
        true_dist.fill_(self.smoothing / (size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(input_logits, true_dist).sum(1).mean() 


class VALOR(VALORModel):
    """ VALOR pretraining """
    def __init__(self, opts):
        super().__init__(opts)
        config = opts
        self.max_generation_len = config.max_generation_len
        self.beam_size  = config.beam_size
        self.beam_size_qa  = config.beam_size_qa
        self.label_smoothing = config.label_smoothing
        if self.label_smoothing > 0:
            self.label_smoothing_loss = LabelSmoothing(self.label_smoothing)
        self.contra_type = config.contra_type
        self.caption_type = config.caption_type
        self.evaluate_ret_text = config.evaluate_ret_text
        self.scst_finetuning = config.scst_finetuning
        self.full_masker = config.full_masker
        self.contra_loss_ratio = config.contra_loss_ratio
        self.fineweight_type = config.fineweight_type  #'none','one','two','three'
        self.use_task_prompt= config.use_task_prompt
        self.late_fusion = config.late_fusion
        if self.scst_finetuning:
            # self.scorer = Scorer()
            self.init_alpah()

                    
        if self.txt_encoder_type.startswith('clip') and self.video_encoder_type.startswith('clip') and opts.init_clip_head:      
            self.contra_head_t = lambda cls_token_t : cls_token_t @ self.clip_model.text_projection
            self.contra_head_v = lambda cls_token_v : cls_token_v @ self.clip_model.visual.proj
            contra_dim = self.clip_model.visual.proj.shape[1]
        else:
            contra_dim = config.contra_dim   
            self.contra_head_t = Contra_head(self.txt_dim, contra_dim)
            self.contra_head_v = Contra_head(self.video_dim, contra_dim)

        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)

        if self.contra_type == 'coarse' and not self.late_fusion:
            self.va_fusion = nn.Linear(2*contra_dim, contra_dim)

        if self.contra_type == 'fine':
            self.text_fine_weight = nn.Sequential(
                    nn.Linear(contra_dim, contra_dim), nn.ReLU(inplace=True),
                    nn.Linear(contra_dim, 1))
            self.video_fine_weight = nn.Sequential(
                    nn.Linear(contra_dim, contra_dim), nn.ReLU(inplace=True),
                    nn.Linear(contra_dim, 1))
            self.audio_fine_weight = nn.Sequential(
                    nn.Linear(contra_dim, contra_dim), nn.ReLU(inplace=True),
                    nn.Linear(contra_dim, 1))

            self.fine_weight_mapper = {'text':self.text_fine_weight,
                                    'video':self.video_fine_weight,
                                    'audio':self.audio_fine_weight}
                                    
        self.dual_softmax = config.dual_softmax
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.contra_sync = True


        

    def forward(self, batch, task, compute_loss=True):

        if task.startswith('pt'):
            return self.forward_pt(batch, task, compute_loss=compute_loss)
        elif task.startswith('ret'):
            return self.forward_ret(batch, task, compute_loss=compute_loss)
        elif task.startswith('cap'):
            return self.forward_cap(batch, task, compute_loss=compute_loss)
        elif task.startswith('qa'):
            return self.forward_qa(batch, task, compute_loss=compute_loss)


    def full_mask(self,txt_tokens):
        b,n = txt_tokens.shape 
        txt_tokens = torch.cat((txt_tokens,torch.zeros_like(txt_tokens).fill_(self.text_mask_token)),dim=1)
        labels = -torch.ones_like(txt_tokens)
        labels[:,n:2*n-1][txt_tokens[:,1:n]!=0] = txt_tokens[:,1:n][txt_tokens[:,1:n]!=0]
        return txt_tokens, labels



    def decode_sequence(self, seq):
        N, T = seq.size()
        sents = []
        for n in range(N):
            tokens = []
            for t in range(T):
                ix = seq[n, t].item()
                if ix == self.eos_token:
                    break
                #words.append(i2v[ix])
                tokens.append(ix)
            if self.tokenizer_type == 'bert':
                words = self.tokenizer.convert_ids_to_tokens(tokens)
                sent = ' '.join(words)
                sents.append(sent.replace(' ##', ''))
            elif self.tokenizer_type == 'clip':
                sents.append(self.tokenizer.decode(tokens))
        return sents


    def reward_loss(self, seq, logP, rewards):
        mask = seq !=self.eos_token 
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss



    
    def compute_fine_matrix(self, featA, featB, maskA, maskB, weightA, weightB):
        if featB.shape[0] > 1200:
            batch_size = featA.shape[0] 
            freq = 100
            slices = math.ceil(batch_size / freq)
            output = []
            for i in range(slices):
                output.append(self.compute_fine_matrix_slice(featA[i*freq:i*freq+freq], featB, maskA[i*freq:i*freq+freq], maskB, weightA[i*freq:i*freq+freq], weightB))
            output = torch.cat(output,dim=0)
        else:
            return self.compute_fine_matrix_slice(featA, featB, maskA, maskB, weightA, weightB)
        return output
    
    def compute_fine_matrix_slice(self, featA, featB, maskA, maskB, weightA, weightB):

        weightA.masked_fill_(torch.tensor((1 - maskA), dtype=torch.bool), float("-inf"))
        weightA = torch.softmax(weightA, dim=-1)  # B x N_t

        weightB.masked_fill_(torch.tensor((1 - maskB), dtype=torch.bool), float("-inf"))
        weightB = torch.softmax(weightB, dim=-1)  # B x N_t


        retrieve_logits = torch.einsum('atd,bvd->abtv', [featA, featB])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, maskA])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, maskB])

        A2B_logits = retrieve_logits.max(dim=-1)[0]  # abtv -> abt
        B2A_logits = retrieve_logits.max(dim=-2)[0]  # abtv -> abv

        A2B_logits = torch.einsum('abt,at->ab', [A2B_logits, weightA])
        B2A_logits = torch.einsum('abv,bv->ab', [B2A_logits, weightB])
        score_matrix = (A2B_logits + B2A_logits) / 2.0

        return score_matrix


    def forward_pt(self, batch, task, compute_loss=True):
   
        task_ls = task.split('_')
        mlm_task = [] 
        caption_task = [] 
        contra_task = [] 
        for i in task_ls:
            if 'mlm' in i:
                mlm_task = i.split('%')[1:]
            elif 'caption' in i:
                caption_task = i.split('%')[1:]
            elif 'contra' in i:
                contra_task = i.split('%')[1:]

        if compute_loss:
            loss_dict = {}
        else:
            evaluation_dict = {}


        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']       
        ids = batch['ids']
        

        video_output = None
        audio_output = None
        txt_tokens_contra = None 

        #### forward_single_modality_encoder
        
        
        if 'v' in ''.join(mlm_task+caption_task+contra_task):
            video_output = self.forward_video_encoder(video_pixels)
        if 'a' in ''.join(mlm_task+caption_task+contra_task):
            audio_output = self.forward_audio_encoder(audio_spectrograms)
        if 't' in ''.join(contra_task):
            txt_tokens_contra = self.get_text_tokens(txt_tokens, self.txt_encoder_type)
            batch_size = txt_tokens_contra.shape[0]
            if  self.use_task_prompt:
                task_prompt = self.get_task_prompt('project language in common space',batch_size)  
            else:
                task_prompt = None 
            
            txt_tokens_len = txt_tokens_contra.shape[1]
            txt_output = self.forward_txt_encoder(txt_tokens_contra, task_prompt)
            if  self.use_task_prompt:
                txt_output = txt_output[:,:txt_tokens_len]
            

        if contra_task != []:
            
            #### process contra task
            feat_t = None
            feat_v = None
            feat_a = None
        
            if 't' in ''.join(contra_task):
                txt_pooled = self.pool_text_for_contra(txt_output, txt_tokens_contra)
                feat_t = self.contra_head_t(txt_pooled)
                feat_t = F.normalize(feat_t,dim=-1)
                if compute_loss:
                    feat_t = ddp_allgather_with_grads.apply(feat_t)
                    txt_tokens_contra = ddp_allgather(txt_tokens_contra)
            if 'v' in ''.join(contra_task):
                video_pooled = self.pool_video_for_contra(video_output)
                feat_v = self.contra_head_v(video_pooled)
                feat_v = F.normalize(feat_v,dim=-1)
                if compute_loss:
                    feat_v = ddp_allgather_with_grads.apply(feat_v)
            if 'a' in ''.join(contra_task):
                audio_pooled = self.pool_audio_for_contra(audio_output)
                feat_a = self.contra_head_a(audio_pooled)
                feat_a = F.normalize(feat_a,dim=-1)
                if compute_loss:
                    feat_a = ddp_allgather_with_grads.apply(feat_a)


            
            if compute_loss:
                contra_loss_tva = None 
                contra_loss_tv = None 
                contra_loss_ta = None 
                contra_loss_va = None
                contra_loss_vta = None
                contra_loss_atv = None 
                if self.contra_type == 'fine':
                    if 'tv' in contra_task:
                        maskA = (txt_tokens_contra !=0).long().cuda()
                        maskB = torch.ones(*feat_v.shape[:2]).long().cuda()
                        weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                        weightB = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                        score_matrix_tv = self.compute_fine_matrix(feat_t, feat_v, maskA, maskB, weightA, weightB)
                        contra_loss_tv = self.contrastive_loss(score_matrix_tv).mean()

                    if 'tva' in contra_task:
                        
                        if self.late_fusion:
                            maskt = (txt_tokens_contra !=0).long().cuda()
                            maskv = torch.ones(*feat_v.shape[:2]).long().cuda()
                            maska = torch.ones(*feat_a.shape[:2]).long().cuda()
                            weightt = torch.ones_like(feat_t[:,:,0])
                            weightv = torch.ones_like(feat_v[:,:,0])
                            weighta = torch.ones_like(feat_a[:,:,0])
                            score_matrix_tva = self.compute_fine_matrix(feat_t, feat_v, maskt, maskv, weightt, weightv) + \
                                                self.compute_fine_matrix(feat_t, feat_a, maskt, maska, weightt, weighta)

                        else:
                            feat_va = torch.cat((feat_v,feat_a),dim=1)

                            maskA = (txt_tokens_contra !=0).long().cuda()
                            maskB = torch.ones(*feat_va.shape[:2]).long().cuda()
                            if self.fineweight_type == 'none':
                                weightA = torch.ones_like(feat_t[:,:,0])
                                weightB = torch.cat(torch.ones_like(feat_v[:,:,0]),torch.ones_like(feat_v[:,:,0]),dim=1)
                            else:
                                weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                                weightB = torch.cat((self.fine_weight_mapper['video'](feat_v).squeeze(2), self.fine_weight_mapper['audio'](feat_a).squeeze(2)),dim=1)
                            
                            score_matrix_tva = self.compute_fine_matrix(feat_t, feat_va, maskA, maskB, weightA, weightB)
                        contra_loss_tva = self.contrastive_loss(score_matrix_tva).mean()
                       
                
                    if 'ta' in contra_task:
                        maskA = (txt_tokens_contra !=0).long().cuda()
                        maskB = torch.ones(*feat_a.shape[:2]).long().cuda()
                        weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                        weightB = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                        score_matrix_ta = self.compute_fine_matrix(feat_t, feat_a, maskA, maskB, weightA, weightB)
                        contra_loss_ta = self.contrastive_loss(score_matrix_ta).mean()
                     
                    
                    if 'va' in contra_task:
                        maskA = torch.ones_like(feat_v[:,:,0])
                        maskB = torch.ones_like(feat_a[:,:,0])
                        weightA = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                        weightB = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                        score_matrix_va = self.compute_fine_matrix(feat_v, feat_a, maskA, maskB, weightA, weightB)
                        contra_loss_va = self.contrastive_loss(score_matrix_va).mean()

                    if 'vta' in contra_task:
                        maskA = torch.ones_like(feat_v[:,:,0])
                        maskB = torch.cat(((txt_tokens_contra !=0).long().cuda(), torch.ones_like(feat_a[:,:,0])),dim=1)
                        feat_ta = torch.cat((feat_t,feat_a),dim=1)
                        weightA = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                        weightB = torch.cat((self.fine_weight_mapper['text'](feat_t).squeeze(2),self.fine_weight_mapper['audio'](feat_a).squeeze(2)),dim=1)
                        score_matrix_vta = self.compute_fine_matrix(feat_v, feat_ta, maskA, maskB, weightA, weightB)
                        contra_loss_vta = self.contrastive_loss(score_matrix_vta).mean()

                    if 'atv' in contra_task:
                        maskA = torch.ones_like(feat_a[:,:,0])
                        maskB = torch.cat(((txt_tokens_contra !=0).long().cuda(), torch.ones_like(feat_v[:,:,0])),dim=1)
                        feat_tv = torch.cat((feat_t,feat_v),dim=1)
                        weightA = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                        weightB = torch.cat((self.fine_weight_mapper['text'](feat_t).squeeze(2),self.fine_weight_mapper['video'](feat_v).squeeze(2)),dim=1)
                        score_matrix_atv = self.compute_fine_matrix(feat_a, feat_tv, maskA, maskB, weightA, weightB)
                        contra_loss_atv = self.contrastive_loss(score_matrix_atv).mean()


                elif self.contra_type == 'coarse':

                    if 'tv' in contra_task:
                        score_matrix_tv = torch.matmul(feat_t,feat_v.permute(1,0))
                        contra_loss_tv = self.contrastive_loss(score_matrix_tv).mean()
                      
                    if 'tva' in contra_task:
                        
                        if self.late_fusion:
                            score_matrix_tva = torch.matmul(feat_t,feat_v.permute(1,0)) + torch.matmul(feat_t,feat_a.permute(1,0))
                        else:
                            feat_va = F.normalize(self.va_fusion(torch.cat((feat_v,feat_a),dim=-1)),dim=-1)
                            score_matrix_tva = torch.matmul(feat_t,feat_va.permute(1,0))
                        contra_loss_tva = self.contrastive_loss(score_matrix_tva).mean()
                       
                
                    if 'ta' in contra_task:
                        
                        score_matrix_ta = torch.matmul(feat_t,feat_a.permute(1,0))
                        contra_loss_ta = self.contrastive_loss(score_matrix_ta).mean()
                    
                lo=[]
                for i in (contra_loss_tva,contra_loss_tv,contra_loss_ta,contra_loss_va,contra_loss_vta,contra_loss_atv):
                    if i is not None:
                        lo.append(i)
                        
                loss_dict['contra_loss'] = sum(lo)/len(lo) * self.contra_loss_ratio
            else:
                evaluation_dict = {}
                evaluation_dict['feat_t'] = feat_t
                evaluation_dict['feat_v'] = feat_v
                evaluation_dict['feat_a'] = feat_a
                evaluation_dict['txt_tokens'] = txt_tokens_contra

        
        txt_tokens = self.get_text_tokens(txt_tokens, self.multimodal_encoder_type)
        
        if video_output is not None:
            video_input = self.get_multimodal_forward_input_video(video_output)                                           
        if audio_output is not None:
            audio_input = self.get_multimodal_forward_input_audio(audio_output)
        batch_size = txt_tokens.shape[0]
 

        if caption_task!=[]:
            caption_loss_tva = None 
            caption_loss_tv = None 
            caption_loss_ta = None

            if self.caption_type == 'unimlm':
                if self.full_masker:
                    txt_input, txt_labels = self.full_mask(txt_tokens)
                else:
                    txt_input, txt_labels = self.text_masker(txt_tokens,0.6)
            elif self.caption_type == 'lm':
                txt_input = txt_tokens
                txt_lens = txt_tokens.shape[1]
                txt_labels =  torch.zeros_like(txt_tokens)
                txt_labels[:,:txt_lens-1] = txt_tokens[:,1:]
                txt_labels[txt_labels==0] = -1


            if 'tva' in caption_task:
                task_prompt = self.get_task_prompt('describe the video with natural language', batch_size) if self.use_task_prompt else None 
                caption_output_tva = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, audio_input, casual = True)
                caption_output_tva = caption_output_tva[:, :txt_input.shape[1], :]
                caption_output_tva = caption_output_tva[txt_labels != -1]
                caption_scores_tva = self.cls(caption_output_tva)
                if compute_loss:
                    caption_loss_tva =  F.cross_entropy(caption_scores_tva, txt_labels[txt_labels != -1])    
                else:
                    evaluation_dict['caption_scores_tva'] = caption_scores_tva 


         
            if 'tv' in caption_task:
                task_prompt = self.get_task_prompt('describe the video with natural language', batch_size) if self.use_task_prompt else None 
                caption_output_tv = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, None, casual = True)
                caption_output_tv = caption_output_tv[:, :txt_tokens.shape[1], :]
                caption_output_tv = caption_output_tv[txt_labels != -1]
                caption_scores_tv = self.cls(caption_output_tv)
                if compute_loss:
                    caption_loss_tv =  F.cross_entropy(caption_scores_tv, txt_labels[txt_labels != -1])    
                else:
                    evaluation_dict['caption_scores_tv'] = caption_scores_tv

            if 'ta' in caption_task:

                task_prompt = self.get_task_prompt('describe the video with natural language', batch_size) if self.use_task_prompt else None 
                caption_output_ta = self.forward_multimodal_encoder(txt_input, task_prompt, None, audio_input, casual = True)
                caption_output_ta = caption_output_ta[:, :txt_tokens.shape[1], :]
                caption_output_ta = caption_output_ta[txt_labels != -1]
                caption_scores_ta = self.cls(caption_output_ta)
                if compute_loss:
                    caption_loss_ta =  F.cross_entropy(caption_scores_ta, txt_labels[txt_labels != -1])     
                else:
                    evaluation_dict['caption_scores_ta'] = caption_scores_ta

            if compute_loss:
                lo=[]
                for i in (caption_loss_tva,caption_loss_tv,caption_loss_ta):
                    if i is not None:
                        lo.append(i)
                        
                loss_dict['caption_loss'] = sum(lo)/len(lo)
            if not compute_loss:
                evaluation_dict['txt_labels_caption'] = txt_labels

        if mlm_task!=[]:
            mlm_loss_tva = None 
            mlm_loss_tv = None 
            mlm_loss_ta = None

            txt_input, txt_labels = self.text_masker(txt_tokens,0.15)


            if 'tva' in mlm_task:
                task_prompt = self.get_task_prompt('predict masked tokens with visual and audio cues', batch_size)
                mlm_output_tva = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, audio_input, casual = False)
                mlm_output_tva = mlm_output_tva[:, :txt_tokens.shape[1], :]
                mlm_output_tva = mlm_output_tva[txt_labels != -1]
                mlm_scores_tva = self.cls(mlm_output_tva)
                if compute_loss:
                    mlm_loss_tva =  F.cross_entropy(mlm_scores_tva, txt_labels[txt_labels != -1])    
                else:
                    evaluation_dict['mlm_scores_tva'] = mlm_scores_tva

           

            if 'tv' in mlm_task:
                task_prompt = self.get_task_prompt('predict masked tokens with visual cues', batch_size)
                mlm_output_tv = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, None, casual = False)
                mlm_output_tv = mlm_output_tv[:, :txt_tokens.shape[1], :]
                mlm_output_tv = mlm_output_tv[txt_labels != -1]
                mlm_scores_tv = self.cls(mlm_output_tv)
                if compute_loss:
                    mlm_loss_tv =  F.cross_entropy(mlm_scores_tv, txt_labels[txt_labels != -1])    
                else:
                    evaluation_dict['mlm_scores_tv'] = mlm_scores_tv

            if 'ta' in mlm_task:
                task_prompt = self.get_task_prompt('predict masked tokens with audio cues', batch_size)
                mlm_output_ta = self.forward_multimodal_encoder(txt_input, task_prompt, None, audio_input, casual = False)
                mlm_output_ta = mlm_output_ta[:, :txt_tokens.shape[1], :]
                mlm_output_ta = mlm_output_ta[txt_labels != -1]
                mlm_scores_ta = self.cls(mlm_output_ta)
                if compute_loss:
                    mlm_loss_ta =  F.cross_entropy(mlm_scores_ta, txt_labels[txt_labels != -1])    
                else:
                    evaluation_dict['mlm_scores_ta'] = mlm_scores_ta

            if compute_loss:
                lo=[]
                for i in (mlm_loss_tva,mlm_loss_tv,mlm_loss_ta):
                    if i is not None:
                        lo.append(i)
                        
                loss_dict['mlm_loss'] = sum(lo)/len(lo)
            
            if not compute_loss:
                evaluation_dict['txt_labels_mlm'] = txt_labels


        if compute_loss:
            return loss_dict
        else:
            return evaluation_dict


    def forward_ret(self, batch, task, compute_loss=True):
        task = task.split('%')[1:]
        
        #### process contra task
        feat_t = None
        feat_v = None
        feat_a = None
        txt_tokens = None 



        if 't' in ''.join(task):
            txt_tokens = batch['txt_tokens']
            txt_tokens = self.get_text_tokens(txt_tokens, self.txt_encoder_type)

            batch_size = txt_tokens.shape[0]
            if  self.use_task_prompt:
                task_prompt = self.get_task_prompt('project language in common space',batch_size)  
            else:
                task_prompt = None 

            txt_tokens_len = txt_tokens.shape[1]
            txt_output = self.forward_txt_encoder(txt_tokens,task_prompt)
            if  self.use_task_prompt:
                txt_output = txt_output[:,:txt_tokens_len]

            txt_output = self.pool_text_for_contra(txt_output, txt_tokens)
            feat_t = self.contra_head_t(txt_output) 
            feat_t = F.normalize(feat_t,dim=-1)
            if compute_loss:
                feat_t = ddp_allgather_with_grads.apply(feat_t)
                txt_tokens = ddp_allgather(txt_tokens)

        if 'v' in ''.join(task):
            video_pixels = batch['video_pixels']
            video_output  = self.forward_video_encoder(video_pixels)
            video_output = self.pool_video_for_contra(video_output)
            feat_v = self.contra_head_v(video_output)
            feat_v = F.normalize(feat_v,dim=-1)
            if compute_loss:
                feat_v = ddp_allgather_with_grads.apply(feat_v)

        if 'a' in ''.join(task):
            audio_spectrograms = batch['audio_spectrograms']
            audio_output  = self.forward_audio_encoder(audio_spectrograms)
            audio_output = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output) 
            feat_a = F.normalize(feat_a,dim=-1)
            if compute_loss:
                feat_a = ddp_allgather_with_grads.apply(feat_a)
           

        if compute_loss:
            contra_loss_tv = None 
            contra_loss_ta = None 
            contra_loss_tva = None 
            contra_loss_va = None
            contra_loss_vta = None
            contra_loss_atv = None 
            loss_dict = {}
            if self.contra_type == 'fine':
                if 'tv' in task:
                    maskA = (txt_tokens !=0).long().cuda()
                    maskB = torch.ones(*feat_v.shape[:2]).long().cuda()
                    weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                    weightB = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                    score_matrix_tv = self.compute_fine_matrix(feat_t, feat_v, maskA, maskB, weightA, weightB)
                    contra_loss_tv = self.contrastive_loss(score_matrix_tv).mean()
                    
                if 'tva' in task:
                    if self.late_fusion:
                        maskt = (txt_tokens !=0).long().cuda()
                        maskv = torch.ones(*feat_v.shape[:2]).long().cuda()
                        maska = torch.ones(*feat_a.shape[:2]).long().cuda()
                        weightt = torch.ones_like(feat_t[:,:,0])
                        weightv = torch.ones_like(feat_v[:,:,0])
                        weighta = torch.ones_like(feat_a[:,:,0])
                        score_matrix_tva = self.compute_fine_matrix(feat_t, feat_v, maskt, maskv, weightt, weightv) + \
                                            self.compute_fine_matrix(feat_t, feat_a, maskt, maska, weightt, weighta)

                    else:

                        feat_va = torch.cat((feat_v,feat_a),dim=1)
                        maskA = (txt_tokens !=0).long().cuda()
                        maskB = torch.ones(*feat_va.shape[:2]).long().cuda()
                        if self.fineweight_type == 'none':
                            weightA = torch.ones_like(feat_t[:,:,0])
                            weightB = torch.cat((torch.ones_like(feat_v[:,:,0]),torch.ones_like(feat_a[:,:,0])),dim=1)
                        else:
                            weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                            weightB = torch.cat((self.fine_weight_mapper['video'](feat_v).squeeze(2), self.fine_weight_mapper['audio'](feat_a).squeeze(2)),dim=1)
                        score_matrix_tva = self.compute_fine_matrix(feat_t, feat_va, maskA, maskB, weightA, weightB)
                       
                    contra_loss_tva = self.contrastive_loss(score_matrix_tva).mean()
                 
                            
                
                if 'ta' in task:
                    maskA = (txt_tokens !=0).long().cuda()
                    maskB = torch.ones(*feat_a.shape[:2]).long().cuda()
                    weightA = self.fine_weight_mapper['text'](feat_t).squeeze(2)
                    weightB = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                    score_matrix_ta = self.compute_fine_matrix(feat_t, feat_a, maskA, maskB, weightA, weightB)
                    contra_loss_ta = self.contrastive_loss(score_matrix_ta).mean()
                   
                if 'va' in task:
                        maskA = torch.ones_like(feat_v[:,:,0])
                        maskB = torch.ones_like(feat_a[:,:,0])
                        weightA = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                        weightB = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                        score_matrix_va = self.compute_fine_matrix(feat_v, feat_a, maskA, maskB, weightA, weightB)
                        contra_loss_va = self.contrastive_loss(score_matrix_va).mean()

                if 'vta' in task:
                    maskA = torch.ones_like(feat_v[:,:,0])
                    maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_a[:,:,0])),dim=1)
                    feat_ta = torch.cat((feat_t,feat_a),dim=1)
                    weightA = self.fine_weight_mapper['video'](feat_v).squeeze(2)
                    weightB = torch.cat((self.fine_weight_mapper['text'](feat_t).squeeze(2),self.fine_weight_mapper['audio'](feat_a).squeeze(2)),dim=1)
                    score_matrix_vta = self.compute_fine_matrix(feat_v, feat_ta, maskA, maskB, weightA, weightB)
                    contra_loss_vta = self.contrastive_loss(score_matrix_vta).mean()

                if 'atv' in task:
                    maskA = torch.ones_like(feat_a[:,:,0])
                    maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_v[:,:,0])),dim=1)
                    feat_tv = torch.cat((feat_t,feat_v),dim=1)
                    weightA = self.fine_weight_mapper['audio'](feat_a).squeeze(2)
                    weightB = torch.cat((self.fine_weight_mapper['text'](feat_t).squeeze(2),self.fine_weight_mapper['video'](feat_v).squeeze(2)),dim=1)
                    score_matrix_atv = self.compute_fine_matrix(feat_a, feat_tv, maskA, maskB, weightA, weightB)
                    contra_loss_atv = self.contrastive_loss(score_matrix_atv).mean()

            elif self.contra_type == 'coarse':
    
                if 'tv' in task:
                    score_matrix_tv = torch.matmul(feat_t,feat_v.permute(1,0))
                    contra_loss_tv = self.contrastive_loss(score_matrix_tv).mean()

                if 'tva' in task:
                    if self.late_fusion:
                        score_matrix_tva = torch.matmul(feat_t,feat_v.permute(1,0)) + torch.matmul(feat_t,feat_a.permute(1,0))
                    else:
                        feat_va = F.normalize(self.va_fusion(torch.cat((feat_v,feat_a),dim=-1)),dim=-1)
                        score_matrix_tva = torch.matmul(feat_t,feat_va.permute(1,0))
                    contra_loss_tva = self.contrastive_loss(score_matrix_tva).mean()
 
                if 'ta' in task:
                    
                    score_matrix_ta = torch.matmul(feat_t,feat_a.permute(1,0))
                    contra_loss_ta = self.contrastive_loss(score_matrix_ta).mean()


                    
            lo=[]
            for i in (contra_loss_tva,contra_loss_tv,contra_loss_ta,contra_loss_va,contra_loss_vta,contra_loss_atv):
                if i is not None:
                    lo.append(i)
                    
            loss_dict['contra_loss'] = sum(lo)/len(lo) 
            return loss_dict 

        else:
            evaluation_dict = {}
            evaluation_dict['feat_t'] = feat_t
            evaluation_dict['feat_v'] = feat_v
            evaluation_dict['feat_a'] = feat_a
            evaluation_dict['txt_tokens'] = txt_tokens

            return evaluation_dict

    def forward_cap(self, batch, task, compute_loss=True):
        task = task.split('%')[1:]
        batch['txt_tokens'] = self.get_text_tokens(batch['txt_tokens'], self.multimodal_encoder_type)
        
        if compute_loss:
            if self.scst_finetuning:
                return self.forward_cap_scst(batch, task)
            else:
                return self.forward_cap_single(batch, task, compute_loss=True)
        else:
            return self.generate_cap(batch, task)

   


    def process_scst(self,seq):
        N, T = seq.size()
        sents = []
        for n in range(N):
            tokens = []
            for t in range(T):
                ix = seq[n, t].item()
                if ix == self.eos_token:
                    break
                tokens.append(ix)
            sents.append(tokens)
        return sents

    def forward_cap_scst(self, batch, task):

        loss_dict = {}
        batch_ids = batch['ids']
        self.eval()
        with torch.no_grad():
            evaluation_dict_greedy = self.generate_cap(batch, task, mode='greedy')  ### compute  reward baseline
        self.train()
        evaluation_dict_sample = self.generate_cap(batch, task, mode='sample')  ### compute  reward baseline

        
        if 'tv' in task:
            generated_sequences_t_v_greedy = self.process_scst(evaluation_dict_greedy['generated_sequences_t_v'])
            generated_sequences_t_v_sample = self.process_scst(evaluation_dict_sample['generated_sequences_t_v'])
            logprobs_t_v_sample = evaluation_dict_sample['logprobs_t_v'] 

            reward_greedy = self.scorer(batch_ids, generated_sequences_t_v_greedy)
            reward_sample = self.scorer(batch_ids, generated_sequences_t_v_sample)

            self.update_alpha(reward_sample, reward_greedy)
            rewards = reward_sample - reward_greedy * self.get_alpha()
            rewards = torch.from_numpy(rewards).float().cuda()
            caption_loss_tv = self.reward_loss(evaluation_dict_sample['generated_sequences_t_v'], logprobs_t_v_sample, rewards)    
            loss_dict['caption_loss_tv'] = caption_loss_tv
           
        if 'tva' in task:
            generated_sequences_t_va_greedy = self.process_scst(evaluation_dict_greedy['generated_sequences_t_va'])
            generated_sequences_t_va_sample = self.process_scst(evaluation_dict_sample['generated_sequences_t_va'])
            logprobs_t_va_sample = evaluation_dict_sample['logprobs_t_va'] 
            reward_greedy = self.scorer(batch_ids, generated_sequences_t_va_greedy)
            reward_sample = self.scorer(batch_ids, generated_sequences_t_va_sample)

            self.update_alpha(reward_sample, reward_greedy)
            rewards = reward_sample - reward_greedy * self.get_alpha()
            rewards = torch.from_numpy(rewards).float().cuda()
            caption_loss_tva = self.reward_loss(evaluation_dict_sample['generated_sequences_t_va'], logprobs_t_va_sample, rewards)    
            loss_dict['caption_loss_tva'] = caption_loss_tva

        if 'ta' in task:
            generated_sequences_t_a_greedy = self.process_scst(evaluation_dict_greedy['generated_sequences_t_a'])
            generated_sequences_t_a_sample = self.process_scst(evaluation_dict_sample['generated_sequences_t_a'])
            logprobs_t_a_sample = evaluation_dict_sample['logprobs_t_a'] 
            reward_greedy = self.scorer(batch_ids, generated_sequences_t_a_greedy)
            reward_sample = self.scorer(batch_ids, generated_sequences_t_a_sample)
            self.update_alpha(reward_sample, reward_greedy)
            rewards = reward_sample - reward_greedy * self.get_alpha()
            rewards = torch.from_numpy(rewards).float().cuda()
            caption_loss_ta = self.reward_loss(evaluation_dict_sample['generated_sequences_t_a'], logprobs_t_a_sample, rewards)    
            loss_dict['caption_loss_ta'] = caption_loss_ta

        return loss_dict


    def forward_cap_single(self, batch, task, cache=None, compute_loss=True, cache_first=False):

        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']

        
        if compute_loss:
            caption_loss_tva = None
            caption_loss_tv = None 
            caption_loss_ta = None 
            loss_dict = {}
            if self.caption_type == 'unimlm':
                if self.full_masker:
                    txt_tokens, txt_labels_caption = self.full_mask(txt_tokens)
                else:
                    txt_tokens, txt_labels_caption = self.text_masker(txt_tokens,0.6)
            elif self.caption_type == 'lm':
                txt_lens = txt_tokens.shape[1]
                txt_labels_caption =  torch.zeros_like(txt_tokens)
                txt_labels_caption[:,:txt_lens-1] = txt_tokens[:,1:]
                txt_labels_caption[txt_labels_caption==0] = -1
            
            txt_input = txt_tokens
            video_input = None 
            audio_input = None 

            if 'v' in ''.join(task):
                video_output = self.forward_video_encoder(video_pixels)
                video_input = self.get_multimodal_forward_input_video(video_output)  
                                                     
            if 'a' in ''.join(task):
                audio_output = self.forward_audio_encoder(audio_spectrograms)
                audio_input = self.get_multimodal_forward_input_audio(audio_output)

                        
            batch_size = txt_tokens.shape[0]

            if 'tva' in task:
                task_prompt = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
                caption_output = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, audio_input, casual=True,full_masker=self.full_masker)
                caption_output_txt = caption_output[:, :txt_input.shape[1], :]
                masked_output = caption_output_txt[txt_labels_caption != -1]
                prediction_scores_caption = self.cls(masked_output)       
                     
                if self.label_smoothing > 0:
                    caption_loss_tva = self.label_smoothing_loss(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1])
                else:
                    caption_loss_tva = F.cross_entropy(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1])           

            if 'tv' in task:
                task_prompt = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
                caption_output = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, None, casual=True,full_masker=self.full_masker)
                caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
                masked_output = caption_output_txt[txt_labels_caption != -1]
                prediction_scores_caption = self.cls(masked_output)   
                if self.label_smoothing > 0:
                    caption_loss_tv = self.label_smoothing_loss(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1])
                else:
                    caption_loss_tv = F.cross_entropy(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1]) 

            if 'ta' in task:
    
                task_prompt = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
                caption_output = self.forward_multimodal_encoder(txt_input, task_prompt, None, audio_input, casual=True,full_masker=self.full_masker)
                caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
                masked_output = caption_output_txt[txt_labels_caption != -1]
                prediction_scores_caption = self.cls(masked_output)   
                if self.label_smoothing > 0:
                    caption_loss_ta = self.label_smoothing_loss(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1])
                else:
                    caption_loss_ta = F.cross_entropy(prediction_scores_caption, txt_labels_caption[txt_labels_caption != -1]) 

           
            lo=[]
            for i in (caption_loss_tva,caption_loss_tv,caption_loss_ta):
                if i is not None:
                    lo.append(i)

            loss_dict['caption_loss'] = sum(lo)/len(lo)
            return loss_dict

        else:
            txt_input = batch['txt_tokens']
            video_input = batch['video_input']
            audio_input = batch['audio_input']
            prompt_input = batch['prompt_input']
            

            if not self.multimodal_use_cross_attn:
                
                caption_output, cache= self.forward_multimodal_encoder(txt_input, prompt_input, \
                                video_input, audio_input, \
                                cache = cache, use_cache = True, casual=True, cache_first=cache_first,
                                cache_type=self.caption_type)
            else:
             
                caption_output = self.forward_multimodal_encoder(txt_input, prompt_input, \
                                video_input, audio_input, \
                                cache = cache, use_cache = False, casual=True, cache_first=cache_first,
                                cache_type=self.caption_type)      
              
            caption_output_txt = caption_output[:, :txt_input.shape[1], :]
            caption_output_txt = caption_output_txt[:, -1]
            prediction_scores_caption = self.cls(caption_output_txt)  
            return prediction_scores_caption, cache


    def get_batch_size(self,batch):
        if 'video_input' in batch and batch['video_input'] is not None:
            batch_size = batch['video_input'].shape[0]
        elif 'audio_input' in batch and batch['audio_input'] is not None:
            batch_size = batch['audio_input'].shape[0]
        elif 'txt_tokens' in batch and batch['txt_tokens'] is not None:
            batch_size = batch['txt_tokens'].shape[0]
        else:
            raise ValueError
        return batch_size
    
    def generate_cap(self, batch, task, mode='none'):
        
        evaluation_dict = {}
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']

    
        video_input = None 
        audio_input = None

        
        if 'v' in ''.join(task):
            video_output = self.forward_video_encoder(video_pixels)
            video_input = self.get_multimodal_forward_input_video(video_output) 
        if 'a' in ''.join(task):
            audio_output = self.forward_audio_encoder(audio_spectrograms)
            audio_input = self.get_multimodal_forward_input_audio(audio_output)
      
            
        if video_input is not None:
            batch_size = video_input.shape[0]
        elif audio_input is not None:
            batch_size = audio_input.shape[0]


        generated_sequences_t_v = None
        generated_sequences_t_va = None
        generated_sequences_t_a = None



        if 'tv' in task:

            batch['video_input'] = video_input
            batch['audio_input'] = None
            batch['prompt_input'] = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
            if self.beam_size >1:
                generated_sequences_t_v = self.decode_beam(batch,'cap')
            else:
                generated_sequences_t_v, logprobs_t_v = self.decode_greedy(batch,'cap')
                evaluation_dict['logprobs_t_v'] = logprobs_t_v
            evaluation_dict['generated_sequences_t_v'] = generated_sequences_t_v


        if 'tva' in task:

            batch['video_input'] = video_input
            batch['audio_input'] = audio_input
            batch['prompt_input'] = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
            if self.beam_size >1:
                generated_sequences_t_va = self.decode_beam(batch,'cap')
            else:
                generated_sequences_t_va, logprobs_t_va = self.decode_greedy(batch,'cap')
                evaluation_dict['logprobs_t_va'] = logprobs_t_va
            evaluation_dict['generated_sequences_t_va'] = generated_sequences_t_va



    
        if 'ta' in task:
    
            batch['video_input'] = None
            batch['audio_input'] = audio_input
            batch['prompt_input'] = self.get_task_prompt('describe the video with natural language',batch_size) if self.use_task_prompt else None 
            if self.beam_size >1:
                generated_sequences_t_a = self.decode_beam(batch,'cap')
            else:
                generated_sequences_t_a, logprobs_t_a = self.decode_greedy(batch,'cap')
                evaluation_dict['logprobs_t_a'] = logprobs_t_a
            evaluation_dict['generated_sequences_t_a'] = generated_sequences_t_a

        return evaluation_dict


    def decode_greedy(self, batch, task, mode='greedy'):
        
        iteration=False
        max_generation_len = self.max_generation_len
        batch_size = self.get_batch_size(batch)
        sents = torch.zeros((batch_size, max_generation_len), dtype=torch.long).fill_(self.eos_token).cuda()
        logprobs = torch.zeros(batch_size, max_generation_len).cuda()
        unfinished = torch.ones(batch_size, dtype=torch.bool).cuda()

        state = None
        cache = {'key':{},'value':{}}
        for t in range(max_generation_len):
            if t==0:
                logits,cache = self.get_logits(batch, state, task, cache, cache_first=True)
            else:
                logits,cache = self.get_logits(batch, state, task, cache, cache_first=False)

            if mode == 'greedy': 
                _ , wt = torch.max(logits, 1)      
            elif mode == 'sample':
                probs_t = F.softmax(logits,dim=1)
                wt = torch.multinomial(probs_t, 1)    
                logP_t = torch.log(probs_t).gather(1, wt)
                logprobs[:,t] = logP_t.view(-1)
 
            else:
                raise NotImplementedError

            wt = wt.view(-1).long()
            if task in ['cap','qa']:
                unfinished = unfinished * (wt != self.eos_token)
                wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * self.eos_token
            sents[:,t] = wt
   
            state = wt.unsqueeze(1) if state is None else torch.cat((state,wt.unsqueeze(1)),dim=1)
        
            if unfinished.sum() == 0:
                break
        
     
        return sents, logprobs  #### return logprobs for scst learning


    def get_logits(self, batch, state, task, cache, cache_first=False):
        batch_size = self.get_batch_size(batch)
        if self.caption_type == 'unimlm':
            masked_tokens = torch.zeros(batch_size,1, dtype = torch.long).cuda().fill_(self.text_mask_token)
            bos_token = torch.zeros(batch_size,1, dtype = torch.long).cuda().fill_(self.bos_token)
            txt_tokens = torch.cat((state,masked_tokens), dim=1 ) if state is not None else masked_tokens
            txt_tokens = torch.cat((bos_token,txt_tokens), dim=1 )
        elif self.caption_type == 'lm':
            bos_token = torch.zeros(batch_size,1, dtype = torch.long).cuda().fill_(self.bos_token)
            txt_tokens = torch.cat((bos_token,state), dim=1 ) if state is not None else bos_token
        
        batch['txt_tokens'] = txt_tokens

       
        if task == 'cap':
            logits,cache = self.forward_cap_single(batch, None, cache, compute_loss = False, cache_first=cache_first)
        elif task == 'qa':
            logits = self.forward_qa_single(batch, None, compute_loss = False )

      
        return logits, cache
     

    def decode_beam(self, batch, task):

        if task in ['cap','qa']:
            max_generation_len = self.max_generation_len       
        else:
            raise NotImplementedError

        beam_size = self.beam_size
        batch_size = self.get_batch_size(batch)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = None
        cache = {'key':{},'value':{},'attn_masks':None}
       
        outputs = []
        for t in range(max_generation_len):

            cur_beam_size = 1 if t == 0 else beam_size
            if t==0:
                word_logits, cache = self.get_logits(batch, state, task, cache, cache_first=True)
                word_logprob = F.log_softmax(word_logits, dim =1 )
                 
            else:
                word_logits, cache = self.get_logits(batch, state, task, cache, cache_first=False)
                word_logprob = F.log_softmax(word_logits, dim =1 )

            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.eos_token).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                #old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # suppress UNK tokens in the decoding
            #candidate_logprob[:, :, candidate_logprob.size(-1) - 1] = -99999

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # for s in range(len(state)):
            #     state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            
            if state is not None:
                state = self._adjust_tensor(batch_size, beam_size, state, selected_beam)
                state = torch.cat((state,selected_words),dim = 1)
                for k in cache['key']:
                    cache['key'][k] = self._adjust_tensor(batch_size, beam_size, cache['key'][k], selected_beam)
                for k in cache['value']:
                    cache['value'][k] = self._adjust_tensor(batch_size, beam_size, cache['value'][k], selected_beam)
                cache['attn_masks'] = self._adjust_tensor(batch_size, beam_size, cache['attn_masks'], selected_beam)

            else:
                state = selected_words
                for k in cache['key']:
                    cache['key'][k] = self.expand_tensor(cache['key'][k], beam_size)
                for k in cache['value']:
                    cache['value'][k] = self.expand_tensor(cache['value'][k], beam_size)
                cache['attn_masks'] = self.expand_tensor(cache['attn_masks'], beam_size)
                
            if t == 0:
                if batch['video_input'] is not None:
                    batch['video_input'] = self.expand_tensor(batch['video_input'], beam_size)
                if batch['audio_input'] is not None:
                    batch['audio_input'] = self.expand_tensor(batch['audio_input'], beam_size)
                if batch['prompt_input'] is not None:
                    batch['prompt_input'] = self.expand_tensor(batch['prompt_input'], beam_size)
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_generation_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_generation_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs

         

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _adjust_tensor(self, batch_size, beam_size, tensor, selected_beam):
        if tensor is None:
            return tensor
        if tensor.dim()==4:        #b,n,c
            b,h,n,c= tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size,h, n,c), 1, \
            selected_beam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size,h,n,c))
            tensor = tensor.reshape(b,h,n,c)
        if tensor.dim()==3:        #b,n,c
            b,n,c= tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size, n,c), 1, \
            selected_beam.unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size,n,c))
            tensor = tensor.reshape(b,n,c)
        
        elif tensor.dim()==2:        #b,n
            b,n = tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size, n), 1, \
            selected_beam.unsqueeze(-1).expand(batch_size, beam_size, n))
            tensor = tensor.reshape(b,n)
        return tensor

    def expand_tensor(self, tensor, size, dim=1):
        if size == 1 or tensor is None:
            return tensor
        tensor = tensor.unsqueeze(dim)
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
        tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
        return tensor


    def forward_qa(self, batch, task, compute_loss=True):
        task = task.split('%')[1:]
        
        batch['question_tokens'] = self.get_text_tokens(batch['question_tokens'], self.multimodal_encoder_type)
        
        if compute_loss:
            batch['txt_tokens'] = self.get_text_tokens(batch['txt_tokens'], self.multimodal_encoder_type)
            return self.forward_qa_single(batch, task, compute_loss=True)
        else:
            return self.generate_qa(batch, task)

    def get_label_tokens(self,label_tokens):
        m,n = len(label_tokens), max([len(i) for i in label_tokens])
        output = torch.zeros((m,n), dtype=torch.long)
        for i in range(len(label_tokens)):
            for j in range(len(label_tokens[i])):
                output[i][j] = label_tokens[i][j]
        output = output.cuda()
        return output


    def forward_qa_single(self, batch, task, compute_loss=True):

        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']

        question_tokens = batch['question_tokens']
        if compute_loss:
            qa_loss_tva = None 
            qa_loss_tv = None 
            qa_loss_ta = None 
            loss_dict = {}
            if self.caption_type == 'unimlm':
                if self.full_masker:
                    txt_tokens, txt_labels_qa = self.full_mask(txt_tokens)
                else:
                    txt_tokens, txt_labels_qa = self.text_masker(txt_tokens,0.99)
            elif self.caption_type == 'lm':
                txt_lens = txt_tokens.shape[1]
                txt_labels_qa =  torch.zeros_like(txt_tokens)
                txt_labels_qa[:,:txt_lens-1] = txt_tokens[:,1:]
                txt_labels_qa[txt_labels_qa==0] = -1

            txt_input = txt_tokens
            video_input = None 
            audio_input = None 
            answer_weights = batch['answer_weights']
            answer_nums = batch['answer_nums']
            tile_feats =  not (np.array(answer_nums) == 1).all() #### for image qa, tile feats for multiple answers 
            if tile_feats:
                questions_expand = []
                for i in range(question_tokens.shape[0]):
                    questions_expand.append( question_tokens[i:i+1].expand(answer_nums[i],-1))
                question_tokens = torch.cat(questions_expand,dim=0)      
            if 'v' in ''.join(task):
                video_output = self.forward_video_encoder(video_pixels)
                video_input = self.get_multimodal_forward_input_video(video_output)   
                if tile_feats:
                    video_input_expand = []
                    for i in range(video_input.shape[0]):
                        video_input_expand.append( video_input[i:i+1].expand(answer_nums[i],-1,-1))
                    video_input = torch.cat(video_input_expand,dim=0)                                        
            if 'a' in ''.join(task):
                audio_output = self.forward_audio_encoder(audio_spectrograms)
                audio_input = self.get_multimodal_forward_input_audio(audio_output)
                if tile_feats:
                    audio_input_expand = []
                    for i in range(audio_input.shape[0]):
                        audio_input_expand.append(audio_input[i:i+1].expand(answer_nums[i],-1,-1))
                    audio_input = torch.cat(audio_input_expand,dim=0)  
            

                        
            batch_size = txt_tokens.shape[0]

            if 'tva' in task:

                if  self.use_task_prompt:
                    task_prompt = self.get_task_prompt('answer the question',batch_size)  
                    task_prompt = task_prompt[:,1:-1]
                    task_prompt = torch.cat((question_tokens[:,0:1],task_prompt,question_tokens[:,1:]),dim=1)
                else:
                    task_prompt = question_tokens 
                qa_output = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, audio_input, casual=True,full_masker=self.full_masker)
                qa_output_txt = qa_output[:, :txt_tokens.shape[1], :]

                prediction_scores_qa = self.cls(qa_output_txt)  
                b,n,c = prediction_scores_qa.shape
                prediction_scores_qa = prediction_scores_qa.reshape(b*n,c)
                txt_labels_qa = txt_labels_qa.reshape(b*n)
                loss = F.cross_entropy(prediction_scores_qa,txt_labels_qa,ignore_index=-1,reduction='none').reshape(b,n)
                txt_labels_qa = txt_labels_qa.reshape(b,n)
                loss = loss.sum(dim=-1)/(txt_labels_qa!=-1).sum(dim=-1)
                if tile_feats:
                    loss = loss * answer_weights
                    qa_loss_tva = loss.sum() / len(answer_nums)
                else: 
                    qa_loss_tva = loss.mean()
   

            if 'tv' in task:
                if self.use_task_prompt:
                    task_prompt = self.get_task_prompt('answer the question',batch_size) 
                    task_prompt = task_prompt[:,1:-1]
                    task_prompt = torch.cat((question_tokens[:,0:1],task_prompt, question_tokens[:,1:]),dim=1)
                else:
                    task_prompt = question_tokens
                qa_output = self.forward_multimodal_encoder(txt_input, task_prompt, video_input, None, casual=True,full_masker=self.full_masker)
                qa_output_txt = qa_output[:, :txt_tokens.shape[1], :]

                prediction_scores_qa = self.cls(qa_output_txt)  
                b,n,c = prediction_scores_qa.shape
                prediction_scores_qa = prediction_scores_qa.reshape(b*n,c)
                txt_labels_qa = txt_labels_qa.reshape(b*n)
                loss = F.cross_entropy(prediction_scores_qa,txt_labels_qa,ignore_index=-1,reduction='none').reshape(b,n)
                txt_labels_qa = txt_labels_qa.reshape(b,n)
                loss = loss.sum(dim=-1)/(txt_labels_qa!=-1).sum(dim=-1)
                if tile_feats:
                    loss = loss * answer_weights
                    qa_loss_tv = loss.sum() / len(answer_nums)
                else: 
                    qa_loss_tv = loss.mean()
     

            if 'ta' in task:
                if self.use_task_prompt:
                    task_prompt = self.get_task_prompt('answer the question',batch_size) if self.use_task_prompt else None 
                    task_prompt = task_prompt[:,1:-1]
                    task_prompt = torch.cat((question_tokens[:,0:1],task_prompt,question_tokens[:,1:]),dim=1)
                else:
                    task_prompt = question_tokens
                qa_output = self.forward_multimodal_encoder(txt_input, task_prompt, None, audio_input, casual=True,full_masker=self.full_masker)
                qa_output_txt = qa_output[:, :txt_tokens.shape[1], :]

                prediction_scores_qa = self.cls(qa_output_txt)  
                b,n,c = prediction_scores_qa.shape
                prediction_scores_qa = prediction_scores_qa.reshape(b*n,c)
                txt_labels_qa = txt_labels_qa.reshape(b*n)
                loss = F.cross_entropy(prediction_scores_qa,txt_labels_qa,ignore_index=-1,reduction='none').reshape(b,n)
                txt_labels_qa = txt_labels_qa.reshape(b,n)
                loss = loss.sum(dim=-1)/(txt_labels_qa!=-1).sum(dim=-1)
                if tile_feats:
                    loss = loss * answer_weights 
                    qa_loss_ta = loss.sum() / len(answer_nums)
                else:
                    qa_loss_ta = loss.mean()
      
  

            
            lo=[]
            for i in (qa_loss_tva,qa_loss_tv,qa_loss_ta):
                if i is not None:
                    lo.append(i)

            loss_dict['qa_loss'] = sum(lo)/len(lo)
            return loss_dict

        else:
            txt_input = batch['txt_tokens']
            video_input = batch['video_input']
            audio_input = batch['audio_input']
            prompt_input = batch['prompt_input']
            
            
            qa_output= self.forward_multimodal_encoder(txt_input, prompt_input, video_input, audio_input, casual=True)
            qa_output_txt = qa_output[:, :txt_tokens.shape[1], :]
            qa_output_txt = qa_output_txt[:, -1]
            prediction_scores_qa = self.cls(qa_output_txt)  
            return prediction_scores_qa



    def generate_qa(self, batch, task):
        
        evaluation_dict = {}
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']

        question_tokens = batch['question_tokens']
        sample_num = batch['sample_num']
    
        if 'v' in ''.join(task):
            video_output = self.forward_video_encoder(video_pixels)
            video_input = self.get_multimodal_forward_input_video(video_output) 
            video_input_expand = []
            for i in range(video_input.shape[0]):
                video_input_expand.append( video_input[i:i+1].expand(sample_num[i],-1,-1))
            video_input = torch.cat(video_input_expand,dim=0)
        if 'a' in ''.join(task):
            audio_output = self.forward_audio_encoder(audio_spectrograms)
            audio_input = self.get_multimodal_forward_input_audio(audio_output)
            audio_input_expand = []
            for i in range(audio_input.shape[0]):
                audio_input_expand.append(audio_input[i:i+1].expand(sample_num[i],-1,-1))
            audio_input = torch.cat(audio_input_expand,dim=0)
        


        batch_size = question_tokens.shape[0]


        generated_answers_t_v = None
        generated_answers_t_va = None
        generated_answers_t_a = None



        if 'tv' in task:

            batch['video_input'] = video_input
            batch['audio_input'] = None
            if  self.use_task_prompt:
                task_prompt = self.get_task_prompt('answer the question',batch_size)  
                task_prompt = task_prompt[:,1:-1]
                task_prompt = torch.cat((question_tokens[:,0:1],task_prompt,question_tokens[:,1:]),dim=1)
            else:
                task_prompt = question_tokens 
            batch['prompt_input'] = task_prompt

            if self.beam_size_qa >1:
                generated_answers_t_v = self.decode_beam(batch,'qa')
            else:
                generated_answers_t_v,_ = self.decode_greedy(batch,'qa')
            evaluation_dict['generated_answers_t_v'] = generated_answers_t_v


        if 'tva' in task:

            batch['video_input'] = video_input
            batch['audio_input'] = audio_input
            if  self.use_task_prompt:
                task_prompt = self.get_task_prompt('answer the question',batch_size)  
                task_prompt = task_prompt[:,1:-1]
                task_prompt = torch.cat((question_tokens[:,0:1],task_prompt,question_tokens[:,1:]),dim=1)
            else:
                task_prompt = question_tokens 
            batch['prompt_input'] = task_prompt

            if self.beam_size_qa >1:
                generated_answers_t_va = self.decode_beam(batch,'qa')
            else:
                generated_answers_t_va,_ = self.decode_greedy(batch,'qa')
            evaluation_dict['generated_answers_t_va'] = generated_answers_t_va


        if 'ta' in task:
    
            batch['video_input'] = None
            batch['audio_input'] = audio_input
            if  self.use_task_prompt:
                task_prompt = self.get_task_prompt('answer the question',batch_size)  
                task_prompt = task_prompt[:,1:-1]
                task_prompt = torch.cat((question_tokens[:,0:1],task_prompt,question_tokens[:,1:]),dim=1)
            else:
                task_prompt = question_tokens 
            batch['prompt_input'] = task_prompt

            if self.beam_size_qa >1:
                generated_answers_t_a = self.decode_beam(batch,'qa')
            else:
                generated_answers_t_a,_ = self.decode_greedy(batch,'qa')
            evaluation_dict['generated_answers_t_a'] = generated_answers_t_a

 

        return evaluation_dict

    def init_alpah(self):
    
        self.alpha_type = 0

        self.total_alpha = 0.7
        self.beta = 1.0
        self.recent_alpha = 0.7
        self.recent_num = 5000
        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

    def update_alpha(self, rewards_sample, rewards_max):

        sample_mean = rewards_sample.mean()
        greedy_mean = rewards_max.mean()

        # total
        self.reward_sample_total += sample_mean
        self.reward_greedy_total += greedy_mean
        self.reward_num += 1
        self.total_alpha = self.reward_sample_total / self.reward_greedy_total

        # recent num
        self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
        self.recent_index += 1
        self.recent_alpha = np.mean(self.recent_alpha_list[:min(self.recent_index, self.recent_num)])

        reward_sample_avg = self.reward_sample_total / self.reward_num
        reward_greedy_avg = self.reward_greedy_total / self.reward_num
        #print("[avg_sample_reward: %.3f] [avg_greedy_reward: %.3f]" % (reward_sample_avg, reward_greedy_avg))

    def get_alpha(self):

        if self.alpha_type == 0:
            temp_alpha = 1.0
        elif self.alpha_type == 1:
            temp_alpha = self.recent_alpha * self.beta
        elif self.alpha_type == 2:
            temp_alpha = self.total_alpha * self.beta
        else:
            raise Exception("Error alpha_type")
        #print("[alpha_type: %d] [total_alpha: %.3f] [recent_alpha: %.3f]" % (self.alpha_type, self.total_alpha, self.recent_alpha))
        return temp_alpha


    def tile(self,x, dim, n_tile):
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device))    
            