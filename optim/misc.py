"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
import ipdb


def build_optimizer(model, opts):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if opts.decoder_lr == -1:
        opts.decoder_lr = opts.learning_rate
    basic_params = []
    basic_params_name = []
    basic_params_no_decay = []
    clip_params_visual = []
    clip_params_name_visual = []
    clip_params_no_decay_visual = []
    clip_params_text = []
    clip_params_name_text = []
    clip_params_no_decay_text = []
    new_params = []
    new_params_name = []
    new_params_no_decay = []
    decoder_params = []
    decoder_params_name = []
    decoder_params_no_decay = []

    for k, v in model.named_parameters():
        if  'clip' in k and  'visual' in k and not any(nd in k for nd in no_decay):
            clip_params_visual.append(v)
            clip_params_name_visual.append(k)
        elif 'clip' in k and  'visual' in k and  any(nd in k for nd in no_decay):
            clip_params_no_decay_visual.append(v)
            clip_params_name_visual.append(k)
        elif  'clip' in k and  not 'visual' in k and not any(nd in k for nd in no_decay):
            clip_params_text.append(v)
            clip_params_name_text.append(k)
        elif 'clip' in k and not 'visual' in k and  any(nd in k for nd in no_decay):
            clip_params_no_decay_text.append(v)
            clip_params_name_text.append(k)      
        elif  'multimodal_encoder.decoder' in k and  not any(nd in k for nd in no_decay):
            decoder_params.append(v)
            decoder_params_name.append(k)
        elif 'multimodal_encoder.decoder' in k  and  any(nd in k for nd in no_decay):
            decoder_params_no_decay.append(v)
            decoder_params_name.append(k)
        elif any(nd in k for nd in opts.new_params_name) and not any(nd in k for nd in no_decay):
            new_params.append(v)
            new_params_name.append(k)
        elif any(nd in k for nd in opts.new_params_name) and any(nd in k for nd in no_decay):
            new_params_no_decay.append(v) 
            new_params_name.append(k)
        elif not any(nd in k for nd in no_decay):
            basic_params.append(v)
            basic_params_name.append(k)
        elif any(nd in k for nd in no_decay):
            basic_params_no_decay.append(v)
            basic_params_name.append(k)


    optimizer_grouped_parameters = [
        {'params': basic_params, 'weight_decay': opts.weight_decay, 'lr': opts.learning_rate},
        {'params': basic_params_no_decay, 'weight_decay': 0.0, 'lr': opts.learning_rate},
        {'params': new_params, 'weight_decay': opts.weight_decay, 'lr': opts.new_lr},
        {'params': new_params_no_decay, 'weight_decay': 0.0, 'lr': opts.new_lr},
        {'params': clip_params_visual, 'weight_decay': opts.weight_decay, 'lr': opts.clip_lr},
        {'params': clip_params_no_decay_visual, 'weight_decay': 0.0, 'lr': opts.clip_lr},
        {'params': clip_params_text, 'weight_decay': opts.weight_decay, 'lr': opts.clip_lr_text},
        {'params': clip_params_no_decay_text, 'weight_decay': 0.0, 'lr': opts.clip_lr_text},
        {'params': decoder_params, 'weight_decay': opts.weight_decay, 'lr': opts.decoder_lr},
        {'params': decoder_params_no_decay, 'weight_decay': 0.0, 'lr': opts.decoder_lr}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')

    for i in optimizer_grouped_parameters:
        i['init_lr'] = i['lr']
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)

    optimizer.new_params_name = new_params_name
    optimizer.new_lr = opts.new_lr
    optimizer.basic_lr = opts.learning_rate
    optimizer.clip_lr_visual = opts.clip_lr
    optimizer.clip_lr_text = opts.clip_lr_text
    optimizer.decoder_lr = opts.decoder_lr
    return optimizer


# def build_optimizer(model, opts):
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer
#                     if not any(nd in n for nd in no_decay)],
#          'weight_decay': opts.weight_decay,
#          'lr': opts.learning_rate},
#         {'params': [p for n, p in param_optimizer
#                     if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0,
#          'lr': opts.learning_rate}
#     ]

#     # currently Adam only
#     if opts.optim == 'adam':
#         OptimCls = Adam
#     elif opts.optim == 'adamax':
#         OptimCls = Adamax
#     elif opts.optim == 'adamw':
#         OptimCls = AdamW
#     else:
#         raise ValueError('invalid optimizer')

#     for i in optimizer_grouped_parameters:
#         i['init_lr'] = i['lr']
#     optimizer = OptimCls(optimizer_grouped_parameters,
#                          lr=opts.learning_rate, betas=opts.betas)

#     return optimizer



def build_optimizer_for_VQA(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = 'vqa_head'
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate * 3},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate * 3}
    ]

    for i in optimizer_grouped_parameters:
        i['init_lr'] = i['lr']
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                          betas=opts.betas)
    return optimizer
