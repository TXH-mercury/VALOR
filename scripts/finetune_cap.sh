basedir=$1

# msrvtt cap

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
--pretrain_dir $basedir \
--config ./config/caption-msrvtt.json \
--output_dir $basedir'/caption-msrvtt-lr9e-6-bs64-epoch5-test10frame-0.05warmup-train6frame'   \
--learning_rate 9e-6  \
--save_best true \
--warmup_ratio 0.05 \
--train_video_sample_num 6 \
--test_video_sample_num 10  \

# --zero_shot \
# --checkpoint PATH/TO/CKPT



# valor32k cap

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-valor32k.json \
# --output_dir $basedir'/cap-valor32k-lr2e-5-bs64'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8  \





# msvd cap 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-msvd.json \
# --output_dir $basedir'/caption-msvd-lr1e-5-bs64-train8test12-flipcrop-lr9e-6-newnew'   \
# --learning_rate 9e-6  \
# --train_video_sample_num 8 \
# --test_video_sample_num 12 \
# --video_transforms 'crop_flip' \
# --train_epoch 100 \
# --save_best true \


#  vatex caption 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-vatex.json \
# --output_dir $basedir'/caption-vatex-lr2e-5-bs64-epoch10'   \
# --learning_rate 2e-5  \
# --train_epoch 100 \
# --train_video_sample_num 8 \
# --test_video_sample_num 20 \
# --save_best true \


#  coco cap

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32712 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-mscoco.json \
# --output_dir $basedir'/caption-mscoco-lr5e-6-bs64-epoch5-res392'   \
# --learning_rate 5e-6  \
# --video_resolution 392 \
# --train_epoch 25 \
# --save_best true \



#  clotho cap

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-clotho.json \
# --output_dir $basedir'/caption-clotho-lr2e-5-bs64'   \
# --learning_rate 2e-5  \
# --save_best true \



#  audiocaps cap

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-audiocaps.json \
# --output_dir $basedir'/caption-audiocaps-lr2e-5-bs64'   \
# --learning_rate 2e-5  \
# --save_best true \




# coco cap scst

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 12711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-mscoco.json \
# --output_dir $basedir'/caption-mscoco-lr2.5e-6-bs64-epoch5-scst-plusbleu-392res'   \
# --learning_rate 2.5e-6  \
# --scst_finetuning true \
# --video_resolution 392 \
# --fp16 false \
# --checkpointing true \
# --checkpoint output/pami/ablation/sota_new_clip_new/caption-mscoco-lr5e-6-bs64-epoch5-res392/ckpt/best_cap%tv--mscoco_cap_tv.pt



# vatex cap scst

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 12711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-vatex.json \
# --output_dir $basedir'/caption-vatex-lr2.5e-6-bs64-epoch5-scst-plusbleu'   \
# --learning_rate 2.5e-6  \
# --scst_finetuning true \
# --train_epoch 50 \
# --train_video_sample_num 8 \
# --test_video_sample_num 20 \
# --first_eval true \
# --fp16 false \
# --checkpointing true \
# --checkpoint output/pami/ablation/sota_new_clip_new/caption-vatex-lr2e-5-bs64-epoch10/ckpt/best_cap%tva%tv--vatex_cap_tva.pt

