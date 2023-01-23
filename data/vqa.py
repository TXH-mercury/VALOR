"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import VALORDataset, TxtMapper
import json
import os
import string 
punctuation = string.punctuation
from pytorch_pretrained_bert import BertTokenizer
import random
from utils.logger import LOGGER
import copy


class TxtMapperForOpenEndedVQA(TxtMapper):
    def __init__(self, txt_dir, opts, max_len, data_type,test_one=False):
        super().__init__(txt_dir, opts, max_len, data_type,test_one)
        self.training = True

    
    def __getitem__(self, id_):
        qa_pairs = self.json_dict[id_]
        if self.training: ####training
            try:
                sample = random.choice(qa_pairs)
            except:
                return None,None,None,None,None
            # while True:
            #     sample = random.choice(qa_pairs)
            #     if sample['answer'] in self.answer_candidate:
            #         break            
            answer_weights = []
            answer_nums = 1
            question, answer = sample['question'], sample['answer']
            question_id = None 
            # print(question)
            # print(answer)
            question_tokens = self.get_single_txt(question)
            if isinstance(answer, str): #### video qa
                answer_tokens = self.get_single_txt(answer, max_len=5)  ####assert answer length <5 for shorter sequences
            elif isinstance(answer, list): #### image qa
                answer_tokens = [self.get_single_txt(ans, max_len=5) for ans in answer]
                answer_weights = sample['answer_weights']
                assert len(answer_tokens) == len(answer_weights)
                answer_nums = len(answer_tokens)
            elif isinstance(answer, int): ### multiple choice 
                answer_tokens =  answer
            
            choice_tokens = None
            if 'choice' in sample:
                choices = sample['choice']
                # print(choices)
                choice_tokens = [self.get_single_txt(choice, max_len=10) for choice in choices]
                
           
            return [question_tokens], answer_tokens, question_id, answer_weights, answer_nums, choice_tokens

        else:           ###testing

            question_tokens = []
            answers = []
            question_ids = None
            choice_tokens = []
            for sample in qa_pairs:
                question, answer = sample['question'], sample['answer']

                q = self.get_single_txt(question)
                question_tokens.append(q)
                answers.append(answer)
                if 'question_id' in sample:
                    if question_ids is None:
                        question_ids = []
                    question_ids.append(sample['question_id'])
                if 'choice' in sample:
                    choices = sample['choice']
  
                    choice_token = [self.get_single_txt(choice, max_len=10) for choice in choices]
                    choice_tokens += choice_token

            return question_tokens, answers, question_ids, None, None, choice_tokens






class VALORQADataset(VALORDataset):
    def __init__(self, ids_path, txt_mapper, video_mapper, audio_mapper,  training):
        super().__init__(ids_path, txt_mapper, video_mapper, audio_mapper, training)
        
    def __getitem__(self, i):
        id_ = self.ids[i]
        
        if self.training:
            question_tokens, answer, question_id, answer_weights, answer_nums,choice_tokens = self.txt_mapper[id_]
            num_samples = 1
            if question_tokens is None: ### without question
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong question, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)
            
          
        else:
            question_tokens, answer, question_id, answer_weights, answer_nums,choice_tokens = self.txt_mapper[id_]
            num_samples = len(answer)
            
           
        if self.video_mapper is not None:
            video_pixels = self.video_mapper[id_]
            if video_pixels is None: ###wrong img/video and needs to resample 
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)
        else:
            video_pixels = None 

        if self.audio_mapper is not None:
            audio_spectrograms = self.audio_mapper[id_]
            if audio_spectrograms is None: ### wrong audio and needs to resample
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)
        else:
            audio_spectrograms = None 



        return id_, question_tokens, answer, question_id, video_pixels, audio_spectrograms, num_samples, answer_weights, answer_nums,choice_tokens



def concat_list(input_list):
    return [k  for i in input_list for k in i]

def valorqa_collate(inputs):
    (ids, question_tokens, answers, question_ids, video_pixels, audio_spectrograms, num_samples,answer_weights, answer_nums,choice_tokens) = map(list, unzip(inputs))


    question_tokens = [ j  for i in question_tokens for j in i]
    question_tokens_collate = {}
    for k in question_tokens[0].keys():  #### bert tokens and clip tokens
        question_tokens_collate[k] = torch.stack([i[k] for i in question_tokens]) 


    if isinstance(question_ids[0],list):  #### testing and 
        question_ids = [j for i in question_ids for j in i]
    
    elif question_ids[0] is None:
        question_ids = None 
    else:
        
        raise NotImplementedError

    choice_tokens_collate=None
    if isinstance(choice_tokens[0],list) and choice_tokens[0]!= [] :

        choice_tokens = [ j  for i in choice_tokens for j in i]
        choice_tokens_collate = {}
        for k in choice_tokens[0].keys():  #### bert tokens and clip tokens
            choice_tokens_collate[k] = torch.stack([i[k] for i in choice_tokens])

    if isinstance(answers[0],int)  :  #### mc and trainign 
        pass
    elif isinstance(answers[0],list) and isinstance(answers[0][0],int):  ###mc and  testing
        answers = [j for i in answers for j in i]
    elif isinstance(answers[0],list) and isinstance(answers[0][0],str):  #### testing
        answers = [j for i in answers for j in i]
    elif isinstance(answers[0], torch.Tensor):  ## training and discriminative
        answers = torch.stack(answers, dim=0)
    elif isinstance(answers[0], dict):  ## training and generative
        answers_collate = {}
        for k in answers[0].keys():  #### bert tokens and clip tokens
            answers_collate[k] = torch.stack([i[k] for i in answers]) 
        answers = answers_collate 
    elif isinstance(answers[0],list) and isinstance(answers[0][0],dict):  #training and image qa
        answers = [ j  for i in answers for j in i]
        answers_collate = {}
        for k in answers[0].keys():  #### bert tokens and clip tokens
            answers_collate[k] = torch.stack([i[k] for i in answers]) 
        answers = answers_collate 
        answer_weights = [ j  for i in answer_weights for j in i]
        answer_weights = torch.tensor(answer_weights)
        #answer_nums = [ j  for i in answer_nums for j in i]

    if video_pixels[0] is not None : #### use video:
        video_pixels = torch.stack(video_pixels, dim=0)
    else:
        video_pixels = None 

    if audio_spectrograms[0] is not None : #### use audio:
        audio_spectrograms = torch.stack(audio_spectrograms, dim=0)
    else:
        audio_spectrograms = None 


 
    batch =   {'ids': ids,
             'txt_tokens': answers,
             'video_pixels': video_pixels,
             'question_ids':question_ids,
             'question_tokens': question_tokens_collate,
             'choice_tokens': choice_tokens_collate,
             'audio_spectrograms': audio_spectrograms,
             'sample_num':num_samples,
             'answer_weights':answer_weights,
             'answer_nums':answer_nums}

    

    return batch
    







