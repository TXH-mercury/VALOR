# VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset
This is the official repository of VALOR which provide training&testing code and pretraining checkpoints. For a comprehensive explaining of VALOR model and dataset, please visit our [project page](https://casia-iva-group.github.io/projects/VALOR).


## Building Environment
VALOR is implemented based on Pytorch. We use pytorch-1.9.0 and cuda-11.1. Other version could be also compatible.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- apex is needed. 
```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- setup packages.

## Download Checkpoints
- [pretrained_weights](https://drive.google.com/file/d/1KyqOzQIzNcL1Q9uEGmDECHfU-8CCd4kk/view?usp=sharing) (BERT,CLIP,VideoSwin).
Put pretrained_weights dir under main path. (VALOR/pretrained_weights)
- [VALOR-base](https://drive.google.com/file/d/1MgwcFqOg-0J7ePRhLIIaHTyPvgWM70d0/view?usp=sharing).
Put VALOR-base  under the output dir. (VALOR/output/VALOR-base)

## Prepare Datasets
VALOR is pretrained and tested on multiple vision-language, audio-language and audiovisual-language datasets. 
e.g. PRETRAIN: VALOR-1M, WebVid-2.5M, CC-3M (VALOR-base)
TEST: VALOR-32K, MSRVTT, MSVD, DiDeMo, LSMDC, ActivityNet, VATEX, AudioCaps, ClothoV1, TGIF-Frame, MSCOCO, VQAV2...
We here take MSRVTT as an example to show the data processing procedures, other datasets take a similar way.

- make dir VALOR/datasets/MSRVTT
- download raw videos from website, and put them in MSRVTT/raw_videos
- extract video frames (.jpg) and audio files (.wav). Utilizing utils/extract_frame_and_wav_multiprocess.py (Note: VALOR use this offline extracted frames and audios for training and testing for it's fast I/O speed. You may adjust to read raw videos via decord library, and need to change VideoMapper and AudioMapper classes in data/data.py.)
- prepare id_files (standardsplit_train_id.json, standardsplit_test_id.json, 1KAsplit_train_id.json, 1KAsplit_test_id.json). The format is List(Str) ['video0', 'video1', ...]. The former two are for video captioning and video qa, while the latter two are for video retrieval.  
- prepare txt_mapper.json. txt_mapper files map videoIDs to its descriptions. Format {'video0':['desc1','desc2',...'desc20']}. For VideoQA task, the format is {'video0':[{'question':'what color is ...?', 'answer':'red'},{'question':'Is the boy ...?', 'answer':'yes'}]}
- prepare caption_annotation.json. This file is used for computing caption metrics. format: [{'video_id':'video0','caption','A boy is ...'}, {'video_id':'video1','caption','A girl is ...'}]

The processed  dataset path should be following:
 ```
    ├── datasets
    │   ├── msrvtt
    │   │   ├── raw_videos
    │   │   │    ├── video0.mp4
    │   │   │    └── video1.mp4
    │   │   ├── frames_fps4
    │   │   │    ├── video0
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   │    └── video1
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   ├── audio_22050hz
    │   │   │    ├── video1.wav
    │   │   │    └── video3.wav
    │   │   ├── standardsplit_train_id.json
    │   │   ├── standardsplit_test_id.json
    │   │   ├── 1KAsplit_train_id.json
    │   │   ├── 1KAsplit_test_id.json
    │   │   ├── txt_mapper.json
    │   │   ├── txt_mapper_1kAsplit_test.json    
    │   │   ├── txt_mapper_vqa.json    
    │   │   └── caption_annotation.json    
```
We provide processed json files for most finetuneing datasets [here](https://drive.google.com/file/d/1pWym3bMNW_WrOZCi5Ls-wKFpaPMbLOio/view?usp=sharing), and you only need to download and extract raw videos of each dataset.


## Finetune  Model
- finetune retrieval tasks
```
sh scripts/finetune_ret.sh $pretrain_path(output/VALOR_base)
```
- finetune captioning tasks
```
sh scripts/finetune_cap.sh $pretrain_path(output/VALOR_base)
```
- finetune QA tasks
```
sh scripts/finetune_qa.sh $pretrain_path(output/VALOR_base)
```
The finetuning output path will be the subdir of $pretrain_path

## Test Model
For example, the cmd for finetuning retrieval model  in scripts/finetune_ret.sh is as follows:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
--pretrain_dir $basedir \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/ret-msrvtt-lr2e-5-bs64-epoch5'   \
--learning_rate 2e-5  \
--train_video_sample_num 4 \
--test_video_sample_num 8  \
--save_best true \
```

if you want to test model, just add following two rows to the cmd:
```
--zero_shot \
--checkpoint $checkpoint_save_path(.pt)
```
## Pretrain Model
```
sh scripts/pretrain.sh
```


## Customize
VALOR's framework is easy to expand new tasks/datasets. what you need to do is 

1. prepare dataset as illustrated above
2. write config file (copy a config file and change 'data_cfg')

- In development stage, you can simply use cmd to overwrite config file. The most important args are :
--learning_rate
--train_batch_size
--train_video_sample_num
--test_video_sample_num
--train_audio_sample_num
--test_audio_sample_num
--video_resolution
--train_epoch
--train_task
--test_task

- To control task and used modality group, you can rewrite train_task by 'task%modality_group1%modality_group2'
For example: finetuning text-to-audio retrieval  'ret%ta' 
             finetuning text-to-video retrieval  'ret%tv' or 'ret%tva' 
             

- Other settings
--fp16 (default: close)
--checkpointing (default: open)




<!-- 
## Citation

If you find this code useful for your research, please consider citing:
```

```

## License

MIT -->
