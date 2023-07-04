# VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset

<div align=center><img src=img/img_radar.png/ width="75%" height="75%"></div>

- This is the official repository of VALOR which provides training&testing code and pretraining checkpoints.
- VALOR-32K dataset (annotation) can be downloaded from  [project page](https://casia-iva-group.github.io/projects/VALOR/download.html). Raw videos can be downloaded from YouTube.
- VALOR-1M will be released after paper is accepted.
- Paper w audio files embeded in PDF can be found on [project page](https://casia-iva-group.github.io/projects/VALOR/download.html).
- We have proposed a   stronger vision-audio-subtitle-text foundation model (VAST), [paper](https://arxiv.org/abs/2305.18500),[github page](https://github.com/TXH-mercury/VAST/tree/main).
- We have proposed a  new strong vision-language pretraining model (COSA), [paper](https://arxiv.org/abs/2306.09085),[github page](https://github.com/TXH-mercury/COSA).


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-retrieval-on-msr-vtt)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-retrieval-on-activitynet)](https://paperswithcode.com/sota/video-retrieval-on-activitynet?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-retrieval-on-vatex)](https://paperswithcode.com/sota/video-retrieval-on-vatex?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/video-retrieval-on-lsmdc?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-captioning-on-vatex-1)](https://paperswithcode.com/sota/video-captioning-on-vatex-1?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-captioning-on-msr-vtt-1)](https://paperswithcode.com/sota/video-captioning-on-msr-vtt-1?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-captioning-on-msvd-1)](https://paperswithcode.com/sota/video-captioning-on-msvd-1?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-question-answering-on-msrvtt-qa)](https://paperswithcode.com/sota/video-question-answering-on-msrvtt-qa?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/tgif-frame-on-tgif-qa)](https://paperswithcode.com/sota/tgif-frame-on-tgif-qa?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/visual-question-answering-on-msvd-qa-1)](https://paperswithcode.com/sota/visual-question-answering-on-msvd-qa-1?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/video-question-answering-on-activitynet-qa)](https://paperswithcode.com/sota/video-question-answering-on-activitynet-qa?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/audio-visual-question-answering-on-music-avqa)](https://paperswithcode.com/sota/audio-visual-question-answering-on-music-avqa?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/text-to-audiovisual-retrieval-on-valor-32k)](https://paperswithcode.com/sota/text-to-audiovisual-retrieval-on-valor-32k?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/audio-visual-captioning-on-valor-32k)](https://paperswithcode.com/sota/audio-visual-captioning-on-valor-32k?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/text-to-audio-retrieval-on-clotho)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-clotho?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/text-to-audio-retrieval-on-audiocaps)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-audiocaps?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/audio-captioning-on-clotho)](https://paperswithcode.com/sota/audio-captioning-on-clotho?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/audio-captioning-on-audiocaps?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/image-captioning-on-coco-captions)](https://paperswithcode.com/sota/image-captioning-on-coco-captions?p=valor-vision-audio-language-omni-perception)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/valor-vision-audio-language-omni-perception/visual-question-answering-on-vqa-v2-test-std)](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2-test-std?p=valor-vision-audio-language-omni-perception)

<div align=center><img src=img/img_model.png/></div>

## Building Environment
- VALOR is implemented based on Pytorch. We use pytorch-1.9.0 and cuda-11.1. Other version could be also compatible.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- build apex. 
```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- install needed packages.
```
sh preinstall.sh
```

## Download Checkpoints
- [pretrained_weights](https://drive.google.com/file/d/1KyqOzQIzNcL1Q9uEGmDECHfU-8CCd4kk/view?usp=sharing) (BERT,CLIP,VideoSwin).
Put pretrained_weights dir under main path. (VALOR/pretrained_weights)
- VALOR models.

| Model   | Pretrained Ckpt | Finetuned Ckpt on MSRVTT-Retrieval | Finetuned Ckpt on MSRVTT-Caption |
|---------|-----------------|------------------------------------|----------------------------------|
| VALOR-B |       [VALOR-base](https://drive.google.com/file/d/1l-G255vTPt6XKMK-Ln42Jz_raGzipL84/view?usp=sharing)      |                [VALOR_base_msr_ret.pt](https://drive.google.com/file/d/1-YrVWKJUwKHTocikN4bvo62Wu78aZhHb/view?usp=sharing)                |               [VALOR_base_msr_cap.pt](https://drive.google.com/file/d/1-mzhin9n9iSCDMjMpAXS8jT2vUHlFN5f/view?usp=sharing)               |
| VALOR-L |       [VALOR-large](https://drive.google.com/file/d/1qFb9ejO-FLUTfZQkW_IFJrFEWyxjs72k/view?usp=sharing)      |                [VALOR_large_msr_ret.pt](https://drive.google.com/file/d/1-XViAyPRovm5WaaN5f1Heh9gPXdqcKBY/view?usp=sharing)                |               [VALOR_large_msr_cap.pt](https://drive.google.com/file/d/1-i_1yfZUMIXbTL8PSM0WtmSguu2eN-kk/view?usp=sharing)               |

Put VALOR-base and VALOR-large under the output dir. (VALOR/output/VALOR-base, VALOR/output/VALOR-large)

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

The processed  dataset path should be as follows:
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
--fp16 (default: True)
--checkpointing (default: False)




## Citation

If you find this code useful for your research, please consider citing:


```
@article{chen2023valor,
  title={VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset},
  author={Chen, Sihan and He, Xingjian and Guo, Longteng and Zhu, Xinxin and Wang, Weining and Tang, Jinhui and Liu, Jing},
  journal={arXiv preprint arXiv:2304.08345},
  year={2023}
}
```

## License

MIT -->
