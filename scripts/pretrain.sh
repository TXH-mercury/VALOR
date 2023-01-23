

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 12226 ./train.py \
--config ./config/pretrain-VALOR-base.json \
--video_encoder_type 'videoswin_base_k600_22k' \
--txt_encoder_type 'bert_base_uncased' \
--output_dir ./output/VALOR_base \
--contra_loss_ratio 1.5







CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 12226 ./train.py \
--config ./config/pretrain-VALOR-large.json \
--output_dir ./output/VALOR_large \
--remove_before_ckpt false \







