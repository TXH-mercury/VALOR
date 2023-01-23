basedir=$1

# msrvtt ret 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
--pretrain_dir $basedir \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/ret-msrvtt-lr2e-5-bs64-epoch5'   \
--learning_rate 2e-5  \
--train_video_sample_num 4 \
--test_video_sample_num 8  \
--save_best true \






#  msvd ret 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-msvd.json \
# --output_dir $basedir'/ret-msvd-lr2e-5-bs64'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --save_best true \


# activitynet ret 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-activitynet.json \
# --output_dir $basedir'/ret-activitynet-lr2e-5-bs64-epoch40-frame8-audioframe4'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 8 \
# --train_audio_sample_num 4 \
# --test_video_sample_num 32  \
# --test_audio_sample_num 8 \
# --checkpointing true \
# --save_best true \



#  valor32k ret

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32721 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-valor32k.json \
# --output_dir $basedir'/ret-valor32k-lr2e-5-bs64-tva'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --train_task 'ret%tva' \
# --test_task 'ret%tva' \


#  didemo ret

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-didemo.json \
# --output_dir $basedir'/ret-didemo-lr2e-5-bs64-epoch5-cliplr2.5e-6-lr9e-6'   \
# --learning_rate 9e-6  \
# --train_video_sample_num 8 \
# --train_audio_sample_num 4 \
# --test_video_sample_num 32 \
# --test_audio_sample_num 8 \
# --train_epoch 30 \
# --save_best true \



#  vatex ret 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-vatex.json \
# --output_dir $basedir'/ret-vatex-50epoch-train4test8'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --save_best true \


#  lsmdc ret 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-lsmdc.json \
# --output_dir $basedir'/ret-lsmdc-lr5e-7-bs64-epoch5-lr2e-5'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --save_best true \
# --train_epoch 5 \



#  mscoco ret res 392 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/fast-retrieval-mscoco.json \
# --output_dir $basedir'/ret-mscoco-lr2e-5-bs1024-epoch15-res392'   \
# --learning_rate 2e-5  \
# --video_resolution 392 \
# --train_batch_size 1024 \
# --train_epoch 75 \
# --checkpointing true \
# --save_best true \
