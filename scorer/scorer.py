import os
import sys
import numpy as np
import pickle
#from lib.config import cfg
import ipdb
from scorer.cider import Cider
from scorer.bleu import Bleu
from cococaption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import json 
import numpy as np 
from collections import defaultdict
from utils.logger import LOGGER
from time import time
# factory = {
#     'CIDEr': Cider,
#     'BLEU': Bleu,
# }


def preprocess_gts(annfile,tokenizer):
    annfile = json.load(open(annfile))
    annfile = annfile['annotations']
    output = defaultdict(list)
    for anno in annfile:
        caption = tokenizer.encode(anno['caption'])
        output[anno['video_id']].append(caption)
    return output


class Scorer(object):
    def __init__(self, annfile, idsfile, tokenizer):
        super(Scorer, self).__init__()
        self.scorers = []
        #self.weights = cfg.SCORER.WEIGHTS
        #self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')

        document_frequency, ref_len = precompute_df_reflen_for_cider(annfile, idsfile, tokenizer)

        self.gts = preprocess_gts(annfile,tokenizer)
        
        #for name in ['CIDEr']:
        self.scorers.append(Cider(document_frequency=document_frequency, ref_len=ref_len))
        self.scorers.append(Bleu())
        self.weights=[1,1]

        self.tokenizer = PTBTokenizer()



    def __call__(self, ids, res):

        # assert set(gts.keys()) == set(res.keys())
        # imgIds = list(gts.keys())
        # time1=time()
        # gts  = self.tokenizer.tokenize(gts)
        # hypo = self.tokenizer.tokenize(res)
        # time2=time()
        # print('tokenize_time:',time2-time1)
        #rewards_info = {}

        gts = [self.gts[i] for i in ids]
        rewards = np.zeros(len(gts))

        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, res)
            
            

            if isinstance(scorer,Bleu):   ###use BLEU4
                #score=score[-1]
                #scores=np.array(scores[-1])
                score=score[-1]
                scores=np.array(scores[-1])
            rewards += self.weights[i] * scores
            #rewards = scores
            #rewards_info[cfg.SCORER.TYPES[i]] = score

        return rewards 
        #return rewards, rewards_info








def precook(words, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    #words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def precompute_df_reflen_for_cider(annfile, idsfile, tokenizer):
    gts = []

    crefs= []
    document_frequency = defaultdict(int)
    ids = json.load(open(idsfile))
    anns = json.load(open(annfile))['annotations']
    id2anns = defaultdict(list)
    for a in anns:
        if a['video_id'] in ids:
            id2anns[a['video_id']].append(tokenizer.encode(a['caption']))  

    # tokenizer = PTBTokenizer()
    # refs = tokenizer.tokenize(id2anns)


    for v in id2anns.values():
        crefs.append(cook_refs(v))
  

    for refs in crefs:
            # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
            document_frequency[ngram] += 1
  

    ref_len = np.log(float(len(crefs)))

    document_lens = len(crefs)
    LOGGER.info(f'document_lens: {document_lens}')

    return document_frequency, ref_len