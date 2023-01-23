import os
from numpy import short
import torch 
import json
import argparse
from data import  TxtMapper,VideoMapper, AudioMapper
from data.data import VALORDataset, valor_collate
from data.vqa import VALORQADataset, valorqa_collate, TxtMapperForOpenEndedVQA
from optim.misc import build_optimizer
from utils.misc import NoOp, parse_with_config, set_random_seed
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from tqdm import tqdm 
from utils.save import ModelSaver
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as apex_DDP

from utils.distributed import DistributedSampler_wopadding
from torch.utils.data import DataLoader
from data import PrefetchLoader
from collections import defaultdict
from apex import amp
import torch.nn.functional as F
from optim import get_lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from time import time

from data import  MetaLoader, PrefetchLoader , AccumMetaLoader
from test import validate
from easydict import EasyDict as edict
from scorer.scorer import Scorer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

#torch.autograd.set_detect_anomaly(True)



def initialize(opts):
    if not os.path.exists(opts.output_dir):
        os.makedirs(os.path.join(opts.output_dir, 'log'), exist_ok=True)
        os.makedirs(os.path.join(opts.output_dir, 'ckpt'), exist_ok=True)
    local_rank = opts.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl') 
    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if dist.get_rank() == 0:
        TB_LOGGER.create(os.path.join(opts.output_dir, 'log'))
        add_log_to_file(os.path.join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
    if opts.test_video_sample_num != -1:
        for d_cfg in opts.data_cfg.val:
            d_cfg.video_sample_num = opts.test_video_sample_num

    if opts.train_video_sample_num != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.video_sample_num = opts.train_video_sample_num

    if opts.test_batch_size != -1:
        for d_cfg in opts.data_cfg.val:
            d_cfg.batch_size = opts.test_batch_size
    
    if opts.train_batch_size != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.batch_size = opts.train_batch_size

    if opts.test_audio_sample_num != -1:
        for d_cfg in opts.data_cfg.val:
            d_cfg.audio_sample_num = opts.test_audio_sample_num


    if opts.train_task != '':
        assert len(opts.data_cfg.train)==1
        opts.data_cfg.train[0].task = opts.train_task

    if opts.test_task != '':
        assert len(opts.data_cfg.val)==1
        opts.data_cfg.val[0].task = opts.test_task

    if opts.video_transforms !='none':
        assert len(opts.data_cfg.train)==1
        assert len(opts.data_cfg.val)==1
        opts.data_cfg.train[0]['datasets'][0]['video_transforms'] = opts.video_transforms
        opts.data_cfg.val[0]['video_transforms'] = opts.video_transforms

    if opts.train_txt_mapper != '':
        assert len(opts.data_cfg.train)==1
        opts.data_cfg.train[0]['datasets'][0]['txt'] = opts.train_txt_mapper

    if opts.train_id != '':
        assert len(opts.data_cfg.train)==1
        opts.data_cfg.train[0]['datasets'][0]['ids_path'] = opts.train_id

    if opts.test_id != '':
        assert len(opts.data_cfg.val)==1
        opts.data_cfg.val[0]['ids_path'] = opts.test_id

    if opts.train_audio_sample_num != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.audio_sample_num = opts.train_audio_sample_num

    if opts.train_epoch != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.epoch = opts.train_epoch

    
  

def load_from_pretrained_dir(opts):

    checkpoint_dir = os.path.os.path.join(opts.pretrain_dir,'ckpt')
    if opts.pretrain_step is not None:
        step = opts.pretrain_step
    else:
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
        
    checkpoint_name = 'model_step_'+str(step)+'.pt'
    ckpt_file = os.path.os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_file, map_location = 'cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    LOGGER.info(f'load_from_pretrained: {ckpt_file}')

    pretrain_cfg = edict(json.load(open(os.path.join(opts.pretrain_dir,'log','hps.json'))))
    ### cover model_cfg 
    cover_cfg=["audio_melbins", "audio_patch_size", "audio_mean", "audio_std",
            "audio_frame_shift", "audio_target_length", "video_encoder_type", 
            "txt_encoder_type", "multimodal_encoder_type", "audio_encoder_type","caption_type",
            "share_txt_and_multimodal","contra_type","multimodal_use_cross_attn", 
           "fineweight_type","has_vafusion_encoder",
            "late_fusion","cross_attn_type","task_pormpt_as_text","use_task_prompt"]
    for k in cover_cfg:
        if k in pretrain_cfg:
            setattr(opts,k,pretrain_cfg[k])

   

    if  'video_frame_embedding' in checkpoint:
        checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num:] = checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num-1].clone()
    if  'audio_frame_embedding' in checkpoint: 
        checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num:] = checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num-1].clone()

    if opts.video_resolution != pretrain_cfg['video_resolution']:
        if opts.video_encoder_type.startswith('clip'):
            vision_width = checkpoint["clip_model.visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["clip_model.visual.conv1.weight"].shape[-1]
            
            grid_size = round((checkpoint["clip_model.visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
            src  = checkpoint["clip_model.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = opts.video_resolution // vision_patch_size
            src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
            src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
            tgt = torch.cat((src_cls,src_oth),dim=0)
            checkpoint["clip_model.visual.positional_embedding"] = tgt
        else:
            pass
    return checkpoint


def load_from_resume(opts):
    ckpt_dir = os.path.join(opts.output_dir,'ckpt')
    previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in  previous_optimizer_state] 
    steps = [ int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)
    
    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state,map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state,map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')



def cover(opts, config, x, y):
    if getattr(opts, x) is not None:
        config[y] =  getattr(opts, x)



def set_parallel_optimizer_and_apex(model, opts, checkpoint_optim):
    device = torch.device("cuda", opts.local_rank)
    model.to(device)

    ### initialize optimizer
    optimizer = build_optimizer(model, opts)
    optimizer.zero_grad()

    ### apex initialize 

    # if opts.amp=='apex':
    model, optimizer = amp.initialize(model, optimizer, enabled=opts.fp16, opt_level='O2')

    # if opts.amp == 'pytorch':
    #     scaler = GradScaler()
    if checkpoint_optim:
        optimizer.load_state_dict(checkpoint_optim)
        del(checkpoint_optim)

    ## parallel
    if not opts.checkpointing:
        model = DDP(model, device_ids=[opts.local_rank], output_device=opts.local_rank, find_unused_parameters=True)
    else:
        pass
    model.train()

    LOGGER.info(f"  basic_lr : {optimizer.basic_lr}")
    LOGGER.info(f"  clip_lr_visual : {optimizer.clip_lr_visual}")
    LOGGER.info(f"  clip_lr_text : {optimizer.clip_lr_text}")
    LOGGER.info(f"  decoder_lr : {optimizer.decoder_lr}")
    LOGGER.info(f"  new_lr : {optimizer.new_lr}")
    LOGGER.info(f"  new_params_name: {optimizer.new_params_name}")

    return model, optimizer

def zero_shot_evaluation(model, test_loader, opts):
    eval_log = validate(model, test_loader, opts, global_step=0, total_step=opts.num_train_steps)
    if dist.get_rank()==0:  
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name +'_' +eval_name 
                # TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                #                 for k, v in metric.items() if not isinstance(v,str)})
                LOGGER.info(f"====-zero-shot evaluation--{eval_name}========\n")
                LOGGER.info(metric)


def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        if 'video_ravg' in metric:
            return 'video_ravg'
        elif 'audio_ravg' in metric:
            return 'audio_ravg'
    elif eval_name.startswith('pt'):
        return None 
        
    else:
        raise NotImplementedError




def conduct_train(model, optimizer, train_loader, val_loaders, LOGGER, TB_LOGGER, opts, start_step=0, verbose_time=False):
  
    if dist.get_rank() == 0:
        pbar = tqdm(total=opts.num_train_steps, initial=start_step)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpt'),remove_before_ckpt=opts.remove_before_ckpt)
    else:
        pbar = NoOp()
        model_saver = NoOp()
        
    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step
    n_gpu = dist.get_world_size()
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    LOGGER.info(f"  Optim : {opts.optim}")
    LOGGER.info(f"  Scheduler : {opts.scheduler}")
    LOGGER.info(f"  Grad_norm : {opts.grad_norm}")
    LOGGER.info(f"  Warmup_ratio : {opts.warmup_ratio}")
    LOGGER.info(f"  Weight_decay: {opts.weight_decay}")
    best_indicator = {}

 
    ### training 
    for step, (name, batch) in enumerate(train_loader):
      
        ndata = train_loader.ndata
        task = name.split('--')[0]
        loss_dict = model(batch, task=task, compute_loss=True)
        loss = sum(list(loss_dict.values()))
        loss_dict['total_loss'] = loss
        loss_dict = {k:v.item() for k,v in loss_dict.items()}

        if opts.dataset_mix_type =='accum' :
            loss = loss / ndata
            delay_unscale = (step+1) % ndata != 0
        else:
            delay_unscale = False

        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            
            scaled_loss.backward()


            if opts.checkpointing:
                works = []
                for p in model.parameters():
                # to speed it up, you can also organize grads to larger buckets to make allreduce more efficient
                    if p.grad is not None:
                        works.append(dist.all_reduce(p.grad, async_op=True))
                for work in works:
                    work.wait()

        if not name in loss_moving_averagetors:
                ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        ####accumulate loss

        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)
    
            
        if (opts.dataset_mix_type =='accum' and (step + 1) % ndata == 0) or opts.dataset_mix_type in ['round-robin','random']:
            global_step += 1
            # learning rate scheduling
            lr_ratio = get_lr_sched(global_step, opts)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['init_lr'] * lr_ratio
                
            TB_LOGGER.add_scalar('lr_ratio', lr_ratio, global_step)
            TB_LOGGER.log_scaler_dict({name: averagetor.val
                                    for name, averagetor in loss_moving_averagetors.items()
                                    if averagetor.val is not None})

            if global_step % 200 == 0:    
                LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
            
            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer), opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)



        if (global_step+1) % opts.valid_steps == 0:
            eval_log = validate(model, val_loaders, opts, global_step, opts.num_train_steps)

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name +'_' +eval_name 
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                                            for k, v in metric.items() if not isinstance(v,str)})
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--==========\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True 
                            else:
                                best_indicator[eval_name] = False 
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}==\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])          
                
                model_saver.save(model, global_step, optimizer,best_indicator, opts.save_best)
        TB_LOGGER.step()

        if global_step >= opts.num_train_steps:
            break
    pbar.close()


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")





def create_train_dataloaders(opts):
    data_cfg = opts.data_cfg.train
    dataloaders = []
    dataloaders_dict={}
    train_steps = []
    loader_names = []
    video_sample_num_ls = []
    audio_sample_num_ls = []
    for d_cfg in data_cfg:
        concate_name = ''
        dataset_ls = []
        # if 'sample_num' in d_

        for dset in d_cfg['datasets']:
            name = dset['name']
            concate_name = concate_name + name if concate_name == '' else concate_name + '_' + name
            assert dset['datatype'] in ['video','image','audio']
            data_type = dset['datatype'] + '_' + name
            ids_path = dset['ids_path']
            
            txt_mapper = None
            video_mapper = None 
            audio_mapper = None


            task_short = [i.split('%')[1:] for i in d_cfg['task'].replace('pt_','').split('_')]
            task_short = [j for i in task_short for j in i]
            task_short = ''.join(task_short)

            if 't' in task_short:
                max_txt_len = d_cfg['max_txt_len']
                if d_cfg['task'].startswith('qa') or d_cfg['task'].startswith('mc'):
                    txt_mapper = TxtMapperForOpenEndedVQA(dset['txt'], opts, max_txt_len, data_type)    
                elif opts.coco_submit or opts.nocaps_submit or opts.vatex_submit:
                    txt_mapper = None
                else:
                    prompt = dset['prompt'] if 'prompt' in dset else None 
                    txt_mapper = TxtMapper(dset['txt'], opts, max_txt_len, data_type, prompt = prompt) 

            if 'v' in task_short:
                video_path = dset['video']
                video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
                video_sample_num_ls.append(video_sample_num)
                video_transforms =  dset.get('video_transforms','none')
                video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num, video_transforms)
            
            if 'a' in task_short:
                audio_path = dset['audio']
                audio_sample_num = d_cfg['audio_sample_num']
                audio_sample_num_ls.append(audio_sample_num)
                audio_mapper = AudioMapper(audio_path, opts, data_type, audio_sample_num)

        
            if d_cfg['task'].startswith('qa'):
                dataset = VALORQADataset(ids_path, txt_mapper, video_mapper, audio_mapper,  training=True)
                collate_fn = valorqa_collate
            else:
                dataset = VALORDataset(ids_path, txt_mapper, video_mapper, audio_mapper, training=True)
                collate_fn = valor_collate
            
            LOGGER.info("Create Dataset {} Success".format(name))
            dataset_ls.append(dataset)
        dataset = ConcatDataset(dataset_ls)
     
        LOGGER.info("Create Dataset {} Success".format(concate_name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 
        epoch = d_cfg['epoch']
        train_steps.append(int((len(dataset) // batch_size) * epoch))
        dataloaders.append(build_dataloader(dataset, collate_fn, True, batch_size, n_workers))
        loader_names.append(f'{task}--{concate_name}')
    
    total_train_steps = sum(train_steps)
    for i in range(len(dataloaders)):
        ratio = train_steps[i]
        dataloaders_dict[loader_names[i]] = (dataloaders[i], ratio)

    n_gpu = dist.get_world_size()
    for name, (loader, ratio) in dataloaders_dict.items():
        epoch = (ratio * loader.batch_size * n_gpu ) // len(loader.dataset)
        LOGGER.info(f" loader {name} , ratio {ratio} , bs_pergpu {loader.batch_size}, n_workers {loader.num_workers}, epoch {epoch}" )

    if opts.dataset_mix_type == 'random':
        meta_loader = MetaLoader(dataloaders_dict,
                                accum_steps=opts.gradient_accumulation_steps,
                                distributed=n_gpu > 1)
        opts.num_train_steps = total_train_steps
    elif opts.dataset_mix_type in ['accum','round-robin']:
        assert opts.gradient_accumulation_steps == 1
        meta_loader = AccumMetaLoader(dataloaders_dict,
                                distributed=n_gpu > 1)
        
        
        
    meta_loader = PrefetchLoader(meta_loader)
    meta_loader.ndata = len(dataloaders_dict)
    opts.valid_steps = opts.num_train_steps // opts.valid_freq -1
    opts.video_sample_num = max(video_sample_num_ls) if len(video_sample_num_ls) > 0 else 0
    opts.audio_sample_num = max(audio_sample_num_ls) if len(audio_sample_num_ls) > 0 else 0

    return meta_loader


def create_val_dataloaders(opts, tokenizer):
    data_cfg = opts.data_cfg.val
    dataloaders = {}
    scorer = None
    for d_cfg in data_cfg:
        name = d_cfg['name']
        assert d_cfg['datatype'] in ['video','image','audio']
        data_type = d_cfg['datatype'] + '_' + name
        ids_path = d_cfg['ids_path']
        
            
        txt_mapper = None
        video_mapper = None 
        audio_mapper = None

        task_short = [i.split('%')[1:] for i in d_cfg['task'].replace('pt_','').split('_')]
        task_short = [j for i in task_short for j in i]
        task_short = ''.join(task_short)

        if 't' in task_short:
            max_txt_len = d_cfg['max_txt_len']
            test_one = getattr(d_cfg,'test_one',False) ### for generation
            if d_cfg['task'].startswith('qa') or d_cfg['task'].startswith('mc'):
                txt_mapper = TxtMapperForOpenEndedVQA(d_cfg['txt'], opts, max_txt_len,data_type,test_one)  
            elif opts.coco_submit or opts.nocaps_submit or opts.vatex_submit:
                txt_mapper = None
            else:
                prompt = d_cfg['prompt'] if 'prompt' in d_cfg else None 
                txt_mapper = TxtMapper(d_cfg['txt'], opts, max_txt_len, data_type,test_one,prompt=prompt)
                
        if 'v' in task_short:
            video_path = d_cfg['video']
            video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
            video_transforms =  d_cfg.get('video_transforms','none')
            video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num,video_transforms)
        
        if 'a' in task_short:
            audio_path = d_cfg['audio']
            audio_sample_num = d_cfg['audio_sample_num']
            audio_mapper = AudioMapper(audio_path, opts, data_type, audio_sample_num)


        if d_cfg['task'].startswith('qa'):
            dataset = VALORQADataset(ids_path, txt_mapper, video_mapper, audio_mapper,  training=False)
            collate_fn = valorqa_collate
        else:
            dataset = VALORDataset(ids_path, txt_mapper, video_mapper, audio_mapper, training=False)
            collate_fn = valor_collate
    

        if d_cfg['task'].startswith('qa') and 'answer_candidate' in d_cfg:
            dataset.answer_candidate = d_cfg['answer_candidate']
        if d_cfg['task'].startswith('cap'):
            dataset.annfile = d_cfg['annfile']
            if opts.scst_finetuning:   #### create scorer for scst finetuning, must only have one train dataset.
                assert len(data_cfg) == 1 
                scorer = Scorer(d_cfg['annfile'],ids_path, tokenizer)
       
        LOGGER.info("Create Dataset {} Success".format(name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 
        loader = build_dataloader(dataset, collate_fn, False, batch_size, n_workers)
        task_name = f'{task}--{name}'
        dataloaders[task_name] = PrefetchLoader(loader)
    return dataloaders,scorer

def build_dataloader(dataset, collate_fn, is_train, batch_size, n_workers=None):
    batch_size = batch_size // dist.get_world_size()
    sampler = DistributedSampler_wopadding(dataset)
    loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                        num_workers=n_workers, pin_memory=True,
                        collate_fn=collate_fn, drop_last=is_train)

    return loader





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_resolution", default=224, type=int)
    parser.add_argument("--audio_melbins", default=64, type=int)
    parser.add_argument("--audio_patch_size", default=16, type=int)
    parser.add_argument("--audio_frame_shift", default=10, type=int)
    parser.add_argument("--audio_target_length", default=512, type=int)
    parser.add_argument("--audio_mean", default=-4.2677393, type=float)
    parser.add_argument("--audio_std", default=4.5689974, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--pretrain_step", default=None, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--clip_lr", default=5e-7, type=float)
    parser.add_argument("--clip_lr_text", default=5e-7, type=float)
    parser.add_argument("--optim", default='adam', choices=['adam', 'adamax', 'adamw'])
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--decoder_lr", default=-1, type=float)
    parser.add_argument("--grad_norm", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument('--resume', action = 'store_true', help='use txt out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--config')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--scheduler', type=str, default='warmup_linear')
    parser.add_argument("--contra_type", type=str, default='fine')
    parser.add_argument("--caption_type", type=str, default='unimlm')
    parser.add_argument("--cross_attn_type", type=str, default='va_concate')
    parser.add_argument("--video_token_sample_num", type=int, default=-1)
    parser.add_argument("--test_video_sample_num", type=int, default=-1)
    parser.add_argument("--test_audio_sample_num", type=int, default=-1)
    parser.add_argument("--max_generation_len", type=int, default=30)
    parser.add_argument("--use_task_prompt", type=str2bool, default=False)
    parser.add_argument("--test_txt_mapper", type=str, default='')
    parser.add_argument("--amp", type=str, default='apex')
    parser.add_argument("--train_txt_mapper", type=str, default='')
    parser.add_argument("--train_id", type=str, default='')
    parser.add_argument("--test_id", type=str, default='')
    parser.add_argument("--init_clip_head", type=str2bool, default=True)
    parser.add_argument("--new_ret", type=str2bool, default=False)
    parser.add_argument("--train_task", type=str, default='')
    parser.add_argument("--test_task", type=str, default='')
    parser.add_argument("--train_audio_sample_num", type=int, default=-1)
    parser.add_argument("--test_batch_size", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=-1)
    parser.add_argument("--late_fusion", type=str2bool, default=False)
    parser.add_argument("--task_pormpt_as_text", type=str2bool, default=False)
    parser.add_argument("--checkpointing", type=str2bool, default=False)
    parser.add_argument("--frozen_multimodal", type=str2bool, default=False)
    parser.add_argument("--frozen_vision", type=str2bool, default=False)
    parser.add_argument("--has_vafusion_encoder", type=str2bool, default=False)
    parser.add_argument("--scst_finetuning", type=str2bool, default=False)
    parser.add_argument("--initial_vision", type=str2bool, default=True)
    parser.add_argument("--save_best", type=str2bool, default=False)
    parser.add_argument("--train_epoch", type=int, default=-1)
    parser.add_argument("--train_video_sample_num", type=int, default=-1)
    parser.add_argument('--video_encoder_type', type=str, default='clip_vit_base_16')
    parser.add_argument('--txt_encoder_type', type=str, default='clip_vit_base_16')
    parser.add_argument('--audio_encoder_type', type=str, default='ast')
    parser.add_argument('--video_transforms', type=str, default='none')
    parser.add_argument('--feature_pooling_type', type=str, default='none')
    parser.add_argument('--multimodal_encoder_type', type=str, default='bert_base_uncased')
    parser.add_argument('--videoswin_timestride', type=int, default=1)
    parser.add_argument('--num_train_steps', type=int, default=0)
    parser.add_argument('--pretrain_dir', type=str, default=None)          
    parser.add_argument('--dual_softmax', type=str2bool, default=False)
    parser.add_argument('--evaluate_ret_text', type=str2bool, default=False)
    parser.add_argument('--first_eval', type=str2bool, default=True)
    parser.add_argument('--loss_reweight', type=str2bool, default=False)
    parser.add_argument('--coco_submit', type=str2bool, default=False)
    parser.add_argument('--vatex_submit', type=str2bool, default=False)
    parser.add_argument('--nocaps_submit', type=str2bool, default=False)
    parser.add_argument('--submit_vizwiz', type=str2bool, default=False)
    parser.add_argument('--loss_mean', type=str2bool, default=False)
    parser.add_argument('--remove_before_ckpt', type=str2bool, default=True)
    parser.add_argument('--contra_loss_ratio', type=float, default=1.0)
    parser.add_argument('--dataset_mix_type', type=str, default='random')
    parser.add_argument('--initial_multimodal', type=str2bool, default=True)
    parser.add_argument('--share_txt_and_multimodal', type=str2bool, default=True)
    parser.add_argument('--use_cache', type=str2bool, default=False)
    parser.add_argument('--multimodal_use_cross_attn', type=str2bool, default=True)
    parser.add_argument('--sample_topk', type=int, default=200)
    parser.add_argument('--cls_dim', type=int, default=0)
    parser.add_argument('--new_lr', type=float, default=0.0)  
    parser.add_argument('--video_reduction', type=str2bool, default=True)
    parser.add_argument('--full_masker', type=str2bool, default=False)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--new_params_name', type=str, default=[], nargs='+') 
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_size_qa', type=int, default=1)
    parser.add_argument('--contra_dim', type=int, default=512)
    parser.add_argument('--label_smoothing', type=float, default=0.0)  
    
    args = parse_with_config(parser)

    return args