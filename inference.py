from model.pretrain import VALOR
import os 
import json
import torch
import torch.nn.functional as F
from torchvision.transforms.transforms import *
from torchvision import transforms
from easydict import EasyDict as edict
import ipdb 
from PIL import Image
import torchaudio
from test import get_model_attr
import argparse
from model.bert_tokenizer import BertTokenizer




parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default=None, type=str)
parser.add_argument("--audio_path", default=None, type=str)
parser.add_argument("--task", default=None, type=str)
parser.add_argument("--question", default=None, type=str)
parser.add_argument("--model_dir", default=None, type=str)


args = parser.parse_args()

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


def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]



def load_from_pretrained_dir(pretrain_dir):
    checkpoint_dir = os.path.os.path.join(pretrain_dir,'ckpt')
   
    checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
    checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
    checkpoint_ls.sort()    
    step = checkpoint_ls[-1]
    
    checkpoint_name = 'model_step_'+str(step)+'.pt'
    ckpt_file = os.path.os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_file, map_location = 'cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}


    pretrain_cfg = edict(json.load(open(os.path.join(pretrain_dir,'log','hps.json'))))
    ### cover model_cfg 
    # cover_cfg=["audio_melbins", "audio_patch_size", "audio_mean", "audio_std",
    #         "audio_frame_shift", "audio_target_length", "video_encoder_type", 
    #         "txt_encoder_type", "multimodal_encoder_type", "audio_encoder_type","caption_type",
    #         "share_txt_and_multimodal","contra_type","multimodal_use_cross_attn", 
    #        "fineweight_type","has_vafusion_encoder",
    #         "late_fusion","cross_attn_type","task_pormpt_as_text","use_task_prompt"]
    # for k in cover_cfg:
    #     if k in pretrain_cfg:
    #         setattr(opts,k,pretrain_cfg[k])

    # opts = 
   

    if  'video_frame_embedding' in checkpoint:
        checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num:] = checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num-1].clone()
    if  'audio_frame_embedding' in checkpoint: 
        checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num:] = checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num-1].clone()

  
    return checkpoint, pretrain_cfg




#pretrain_dir = '/public/chensihan/projects/VALOR/output/VALOR_base/caption-msrvtt-lr9e-6-bs64-epoch5-test10frame-0.05warmup-train6frame'


checkpoint,pretrain_cfg = load_from_pretrained_dir(args.model_dir)

from model.pretrain import VALOR
model = VALOR.from_pretrained(pretrain_cfg,checkpoint)

model.eval().cuda()


#video_path = '/public/chensihan/datasets/music-avqa/all_videos/00000002.mp4'
video_path = args.video_path
video_name = video_path.split('/')[-1].split('.')[0]
output_dir = f'./inference/{video_name}'
fps_frame_dir = os.path.join(output_dir, f"frames_fps1")
os.makedirs(fps_frame_dir, exist_ok=True)
cmd = "ffmpeg -loglevel error -i {} -vsync 0 -f image2 -vf fps=fps={:.02f} -qscale:v 2 {}/frame_%04d.jpg".format(
        video_path, 1, fps_frame_dir)
os.system(cmd)
# Extract Audio
audio_file_path = os.path.join(output_dir,video_name+'.mp4')


cmd = "ffmpeg -i {} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {} -y {}".format(
        video_path, 22050, audio_file_path)
os.system(cmd)





if pretrain_cfg.video_encoder_type.startswith('clip'):
    mean = [0.48145466, 0.4578275, 0.40821073] 
    std  = [0.26862954, 0.26130258, 0.27577711]
else:       
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]



test_transforms = transforms.Compose([Resize((224,224)),
                                            Normalize(mean,std)])





video_pixels = []        
frame_path = fps_frame_dir
frames = os.listdir(frame_path)
frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
sample_num = 8
frames_splited = split(frames,sample_num)    

sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
for i in range(sample_num):
    frame = Image.open(os.path.join(frame_path,sample_idx[i]))
    frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
    video_pixels.append(frame.unsqueeze(0))
video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW

video_pixels = test_transforms(video_pixels)     


 
    



  
        
        

wav_file = audio_file_path
if not os.path.exists(wav_file):
    audio = torch.zeros(self.sample_num, self.melbins, self.target_length)


else:

    target_length = 512
    sample_num=1
    waveform, sr = torchaudio.load(wav_file)

    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                            window_type='hanning', num_mel_bins=pretrain_cfg.audio_melbins, dither=0.0, frame_shift=pretrain_cfg.audio_frame_shift)

                                #### fbank shape :(src_length,64)
    src_length = fbank.shape[0]

    # #### sample 
    output_slices = []

    pad_len = target_length - src_length % target_length
    fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
    total_slice_num = fbank.shape[0] // target_length
    total_slice_num = list(range(total_slice_num))
    total_slice_num = split(total_slice_num, sample_num)


    sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]


    for i in sample_idx:
        output_slices.append(fbank[i*target_length : (i+1)*target_length])

    fbank = torch.stack(output_slices,dim=0).permute(0,2,1)   

        

    ### normalization
    fbank = (fbank - pretrain_cfg.audio_mean) / (pretrain_cfg.audio_std * 2)

    #return fbank.permute(1,0)  ### 128, target_length
 
           

print(video_pixels.shape)
print(fbank.shape)



if args.task.startswith('cap'):
    if args.task == 'cap%tva':


        batch =   {'ids': None,
                    'txt_tokens': None,
                    'video_pixels': video_pixels.unsqueeze(0).cuda(),
                    'audio_spectrograms': fbank.unsqueeze(0).cuda(),
                    'ids_txt':None,
                    'sample_num':None}
                    
        evaluation_dict = model(batch, 'cap%tva', compute_loss=False)

        # print(evaluation_dict)


        sents = evaluation_dict['generated_sequences_t_va']
        sents = get_model_attr(model, 'decode_sequence')(sents.data)

        print(sents)
    
    elif args.task == 'cap%tv':

        batch =   {'ids': None,
                    'txt_tokens': None,
                    'video_pixels': video_pixels.unsqueeze(0).cuda(),
                    'ids_txt':None,
                    'sample_num':None}
                    
        evaluation_dict = model(batch, 'cap%tva', compute_loss=False)

        # print(evaluation_dict)


        sents = evaluation_dict['generated_sequences_t_v']
        sents = get_model_attr(model, 'decode_sequence')(sents.data)

        print(sents)
    
    else:
        raise NotImplementedError
    


elif args.task.startswith('qa'):
    assert args.question is not None
    bert_tokenizer = BertTokenizer("./pretrained_weights/bert-base-uncased-vocab.txt")
    cls_token = bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    sep_token = bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    

    tokenized_text = bert_tokenizer.tokenize(args.question)
    txt_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    txt_tokens = [cls_token] + txt_tokens + [sep_token]  

    txt_tokens = {'bert_tokens':torch.tensor(txt_tokens, dtype=torch.long).unsqueeze(0).cuda()}


    if args.task == 'qa%tva':
        batch =   {'ids': None,
                    'question_tokens': txt_tokens,
                    'audio_spectrograms': fbank.unsqueeze(0).cuda(),
                    'video_pixels': video_pixels.unsqueeze(0).cuda(),
                    'ids_txt':None,
                    'sample_num':[1]}

                        
        evaluation_dict = model(batch, 'qa%tva', compute_loss=False)

        # print(evaluation_dict)


        sents = evaluation_dict['generated_answers_t_va']
        sents = get_model_attr(model, 'decode_sequence')(sents.data)

        print(sents)
    
    elif args.task == 'qa%tv':
        batch =   {'ids': None,
                    'question_tokens': txt_tokens,
                    'video_pixels': video_pixels.unsqueeze(0).cuda(),
                    'ids_txt':None,
                    'sample_num':[1]}

                        
        evaluation_dict = model(batch, 'qa%tv', compute_loss=False)

        # print(evaluation_dict)


        sents = evaluation_dict['generated_answers_t_v']
        sents = get_model_attr(model, 'decode_sequence')(sents.data)

        print(sents)
    
