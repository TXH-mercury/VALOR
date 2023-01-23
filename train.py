from utils.logger import LOGGER, TB_LOGGER
from train_utils import initialize, load_from_pretrained_dir,  conduct_train, \
                                set_parallel_optimizer_and_apex, zero_shot_evaluation, get_args, create_train_dataloaders, load_from_resume, \
                                    create_val_dataloaders, set_dropout
from test import get_model_attr
from model.pretrain import VALOR
import os 
import json
import torch
import torch.nn.functional as F

def main(opts):

    ### init 
    initialize(opts)
    
    checkpoint = {}

    if opts.pretrain_dir is not None:

        checkpoint = load_from_pretrained_dir(opts)
        LOGGER.info("Load from pretrained dir {}".format(opts.pretrain_dir))


    if opts.checkpoint is not None:
        checkpoint =  torch.load(opts.checkpoint, map_location = 'cpu')

        if "clip_model.visual.positional_embedding" in checkpoint:
            if int((checkpoint["clip_model.visual.positional_embedding"].shape[0] - 1) ** 0.5) != opts.video_resolution:
            
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


    if opts.resume:
        checkpoint, checkpoint_optim, start_step = load_from_resume(opts)
    else:
        checkpoint_optim, start_step = None , 0


    
    ### initialize model 
    model = VALOR.from_pretrained(opts,checkpoint)

    ### set parallel, optimizer  and apex 
    model, optimizer = set_parallel_optimizer_and_apex(model, opts, checkpoint_optim)


    ### create datasets and dataloader
    train_loader = create_train_dataloaders(opts)
    val_loaders,scorer = create_val_dataloaders(opts, get_model_attr(model,'tokenizer'))

    if hasattr(model,'module'):
        model.module.scorer = scorer 
    else:
        model.scorer = scorer

    with open(os.path.join(opts.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(opts), writer, indent=4)

    ## evaluation first time 
    if opts.first_eval:

        zero_shot_evaluation(model, val_loaders, opts)
                                           
        if opts.zero_shot:
            return 
     
    #### start training
    conduct_train(model, optimizer, train_loader, val_loaders, LOGGER, TB_LOGGER, opts, start_step = start_step, verbose_time=False)



if __name__ == "__main__":
    args = get_args()
    main(args)
