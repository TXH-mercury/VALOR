"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""

from cProfile import label
import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset

from torchvision.transforms.transforms import *
from torchvision import transforms
import random
from os.path import join 
from decord import VideoReader, AudioReader
# import av
import os
import torchaudio
from PIL import Image
from utils.logger import LOGGER
import ipdb
import matplotlib.pyplot as plt
import string 
from time import time
from typing import List, Tuple, Optional, Dict
from torch import Tensor
import torch.nn.functional as F
punctuation = string.punctuation
import numpy as np
from torchvision.transforms import functional as transform_F
# from torchvision.transforms import RandAugment
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import librosa
# from model.prompt_template import templates



class TxtMapper(object):
    def __init__(self, txt_dir, opts, max_len, data_type,test_one=False,prompt=None):
        self.max_len = max_len
        self.txt_dir = txt_dir
        self.training = True 
        self.txt_encoder_type = opts.txt_encoder_type
        self.multimodal_encoder_type = getattr(opts,'multimodal_encoder_type','bert_base_uncased')
        self.json_dict = json.load(open(txt_dir))
        self.bert_tokenizer = None 
        self.clip_tokenizer = None
        self.data_type = data_type
        self.punctuations = string.punctuation
        self.test_one=test_one

        if self.txt_encoder_type.startswith('bert') or self.multimodal_encoder_type.startswith('bert'):
            from model.bert_tokenizer import BertTokenizer
            if self.multimodal_encoder_type == 'bert_base_chinese':
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            else:
                self.bert_tokenizer = BertTokenizer("./pretrained_weights/bert-base-uncased-vocab.txt")
            self.cls_token = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
            self.sep_token = self.bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
            assert self.cls_token==101
            assert self.sep_token==102
  
        if self.txt_encoder_type.startswith('clip') or self.multimodal_encoder_type.startswith('clip'):
            from model.clip_tokenizer import SimpleTokenizer
            self.clip_tokenizer = SimpleTokenizer()
            self.sot_token = self.clip_tokenizer.encoder["<|startoftext|>"]
            self.eot_token = self.clip_tokenizer.encoder["<|endoftext|>"]

    def __getitem__(self, id_):
        text = self.json_dict[id_]
        if isinstance(text,list):
            if self.training:
                text = random.choice(text)
                output = [self.get_single_txt(text)]
            elif self.test_one:
                text =  text[0]
                output = [self.get_single_txt(text)]
            else:
                output=[]
                for i in text:          
                    output.append(self.get_single_txt(i))
        else:                
            output = [self.get_single_txt(text)]
        return output



    def get_single_txt(self,text,max_len=None):
        # print(text)
        text = self.clean(text) 
        output = {}
        if self.bert_tokenizer is not None:
            tokenized_text = self.bert_tokenizer.tokenize(text)
            txt_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            bert_tokens =self.get_padded_tokens(txt_tokens,'bert',max_len)
            output['bert_tokens'] = bert_tokens


        if self.clip_tokenizer is not None:
            
            txt_tokens = self.clip_tokenizer.encode(text)
            clip_tokens =self.get_padded_tokens(txt_tokens,'clip',max_len)
            output['clip_tokens'] = clip_tokens
    
        return output
    def clean(self, text):
        """remove duplicate spaces, lower and remove punctuations """
        text = ' '.join([i for i in text.split(' ') if i != ''])
        text = text.lower()
        for i in self.punctuations:
            text = text.replace(i,'')
        return text

    def get_padded_tokens(self,txt_tokens, type, max_len=None):
        
        max_len = self.max_len if  max_len is None else max_len
        txt_tokens = txt_tokens[:max_len]

        if type=='bert':
            txt_tokens = [self.cls_token] + txt_tokens + [self.sep_token]  
        elif type=='clip':
            txt_tokens = [self.sot_token] + txt_tokens + [self.eot_token] 

        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(max_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output



    def detokenize(self, ids):
        return  self.tokenizer.convert_ids_to_tokens(ids)


class VideoMapper(object):
    def __init__(self, video_dir, opts, data_type = 'video', sample_num = 4, video_transforms='none'):
        self.video_dir = video_dir
        self.datatype = data_type
        self.frame_syncaug = True
        self.training = True
        self.sample_num = sample_num 
        
        self.resolution = opts.video_resolution

        if opts.video_encoder_type.startswith('clip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        
        LOGGER.info(f'{data_type} mean : {self.mean}')
        LOGGER.info(f'{data_type} std : {self.std}')      
        
        self.video_transforms = video_transforms
        if video_transforms == 'none':
            self.train_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
                
            self.test_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
        elif video_transforms == 'crop_flip':
            self.train_transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                        RandomHorizontalFlip(),
                                                        Normalize(self.mean,self.std)])

            self.test_transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])
                                    
        else:
            raise NotImplementedError

        LOGGER.info(f'{data_type} video_transforms : {video_transforms} ')    
            
    def __getitem__(self, id_):
      
        if  self.datatype.startswith('video'):
            try:
                video_pixels = []        
                frame_path = os.path.join(self.video_dir, id_)
                frames = os.listdir(frame_path)
                frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                sample_num = self.sample_num
                frames_splited = split(frames,sample_num)    
                if self.training:
                    sample_idx = [random.choice(i) for i in frames_splited]
                else:
                    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
                for i in range(sample_num):
                    frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                    frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                    video_pixels.append(frame.unsqueeze(0))
                video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW
                if self.training:
                    video_pixels = self.train_transforms(video_pixels)    
                else:
                    video_pixels = self.test_transforms(video_pixels)     
                return video_pixels

            except Exception as e:
                print(e)
                return None



        elif self.datatype.startswith('image'):
            try:
                
                if self.datatype.startswith('image_vg'):
                    id_, width, height, x, y = id_.split('%')
                    width = int(width.split('width')[1])
                    height = int(height.split('height')[1])
                    x = int(x.split('x')[1])
                    y = int(y.split('y')[1])
                img_path = os.path.join(self.video_dir, id_)
                if not os.path.exists(img_path):
                    img_path += '.jpg'
                if not os.path.exists(img_path):
                    img_path =  img_path.replace('.jpg','.JPEG')
       
                img = Image.open(img_path)
                img = img.convert('RGB')  #### convert 1-channel gray image and 4-channel CMYK image to RGB image
                
                if self.datatype.startswith('image_vg'):
                    img = transform_F.resized_crop(img, y, x, height, width, (height,width))
                
                if self.training:    
                    img = self.train_transforms(img)
                else:
                    img = self.test_transforms(img)

                img = img.unsqueeze(0)
                return img
            except Exception as e:
                return None
        else:
            raise NotImplementedError()

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]




class AudioMapper(object):
    def __init__(self, audio_dir, opts, data_type, sample_num):
        self.audio_dir = audio_dir
        self.melbins = opts.audio_melbins
        self.target_length = opts.audio_target_length
        self.mean = opts.audio_mean
        self.std = opts.audio_std
        self.training = True
        self.frame_shift = opts.audio_frame_shift
        self.sample_num = sample_num
        self.datatype = data_type
        self.audio_encoder_type = opts.audio_encoder_type
        if opts.audio_encoder_type.startswith('pann'):
            self.target_length = 160000
        else:
            self.target_length = 512

        
        

    def __getitem__(self, id_):

        wav_file = os.path.join(self.audio_dir, id_+'.wav')
        if not os.path.exists(wav_file):
            wav_file = wav_file.replace('wav','mkv')
        if not os.path.exists(wav_file):
            return torch.zeros(self.sample_num, self.melbins, self.target_length)
        

        try:
            if self.audio_encoder_type.startswith('ast'):
                   #### has no audio channel, use zero instead
                # LOGGER.info(f'{id_} has no audio file, use zero instead')
                    
                waveform, sr = torchaudio.load(wav_file)

                waveform = waveform - waveform.mean()
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.frame_shift)

                                            #### fbank shape :(src_length,64)
                src_length = fbank.shape[0]

                # #### sample 
                output_slices = []

                pad_len = self.target_length - src_length % self.target_length
                fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
                total_slice_num = fbank.shape[0] // self.target_length
                total_slice_num = list(range(total_slice_num))
                total_slice_num = split(total_slice_num, self.sample_num)
                
                if self.training:
                    sample_idx = [random.choice(i) for i in total_slice_num]
                else:
                    sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]

                
                for i in sample_idx:
                    output_slices.append(fbank[i*self.target_length : (i+1)*self.target_length])
                
                fbank = torch.stack(output_slices,dim=0).permute(0,2,1)   

                    

                ### normalization
                fbank = (fbank - self.mean) / (self.std * 2)

                #return fbank.permute(1,0)  ### 128, target_length
                return fbank
           

        except Exception as e:
            print(e)
            return    
                

class VALORDataset(Dataset):
    def __init__(self, ids_path, txt_mapper, video_mapper, audio_mapper, training):
        self.txt_mapper = txt_mapper
        self.video_mapper = video_mapper
        self.audio_mapper = audio_mapper
        if self.txt_mapper is not None:
            self.txt_mapper.training = training 
        if self.video_mapper is not None:
            self.video_mapper.training = training 
        if self.audio_mapper is not None:
            self.audio_mapper.training = training 
        self.ids = json.load(open(ids_path)) 
        self.idx = list(range(len(self.ids)))
        if self.video_mapper is not None:
            self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        else:
            self.dataset_name = 'none'
        self.training = training
        
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        id_txt = None 
        txt_tokens = None 
        video_pixels = None 
        audio_spectrograms = None 
        num_samples = None

        if self.txt_mapper is not None:
            txt_tokens = self.txt_mapper[id_]
            if self.training:
                id_txt = id_  
                num_samples = 1 
            else:
                id_txt = [id_] * len(txt_tokens)
                num_samples = len(txt_tokens)
        

        if self.video_mapper is not None:
            video_pixels = self.video_mapper[id_]
            if video_pixels is None: ###wrong img/video and needs to resample 
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)


        if self.audio_mapper is not None:   
            audio_spectrograms = self.audio_mapper[id_]
            if audio_spectrograms is None: ### wrong audio and needs to resample
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)


        return id_, txt_tokens, video_pixels, audio_spectrograms, id_txt, num_samples




def valor_collate(inputs):
    

    (ids, txt_tokens, video_pixels, audio_spectrograms, ids_txt, num_samples) = map(list, unzip(inputs))


    if isinstance (ids_txt[0], list) : ### testing:
        ids_txt = [ j  for i in ids_txt for j in i]
    elif  isinstance (ids_txt[0], str) : ### training:
        pass
    elif  ids_txt[0] is None:
        ids_txt = None 

    if txt_tokens[0] is not None:
        txt_tokens = [ j  for i in txt_tokens for j in i]
        txt_tokens_collate = {}
        for k in txt_tokens[0].keys():  #### bert tokens and clip tokens
            txt_tokens_collate[k] = torch.stack([i[k] for i in txt_tokens]) 
    else:
        txt_tokens_collate = None 

    if video_pixels[0] is not None : #### use video:
        video_pixels = torch.stack(video_pixels, dim=0)
    else:
        video_pixels = None 

    if audio_spectrograms[0] is not None : #### use audio:
        audio_spectrograms = torch.stack(audio_spectrograms, dim=0)
    else:
        audio_spectrograms = None 



        

    batch =   {'ids': ids,
             'txt_tokens': txt_tokens_collate,
             'video_pixels': video_pixels,
             'audio_spectrograms': audio_spectrograms,
             'ids_txt':ids_txt,
             'sample_num':num_samples}
    
    return batch

    
