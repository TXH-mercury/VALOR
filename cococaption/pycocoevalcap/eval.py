from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
#from .spice.spice import Spice
from .wmd.wmd import WMD
import ipdb
class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalVideos = []
        self.eval = {}
        self.videoToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        #self.params = {'video_id': coco.getVideoIds()}
        self.params = {'video_id': cocoRes.getVideoIds()}

        #self.Spice = Spice()

    def evaluate(self):
        videoIds = self.params['video_id']
        # videoIds = self.coco.getvideoIds()
        gts = {}
        res = {}
        print('total test num:{}'.format(len(videoIds)))
        for videoId in videoIds:
            gts[videoId] = self.coco.videoToAnns[videoId]
            res[videoId] = self.cocoRes.videoToAnns[videoId]

        #ipdb.set_trace()
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        # import ipdb 
        # ipdb.set_trace()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(self.Spice, "SPICE"),
            #(WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setvideoToEvalvideos(scs, list(gts.keys()), m)
                    #print("%s: %0.1f"%(m, sc*100))
            else:
                self.setEval(score, method)
                self.setvideoToEvalvideos(scores, list(gts.keys()), method)
                #print("%s: %0.1f"%(method, score*100))
        self.setEvalvideos()

    def setEval(self, score, method):
        self.eval[method] = score

    def setvideoToEvalvideos(self, scores, videoIds, method):
        for videoId, score in zip(videoIds, scores):
            if not videoId in self.videoToEval:
                self.videoToEval[videoId] = {}
                self.videoToEval[videoId]["video_id"] = videoId
            self.videoToEval[videoId][method] = score

    def setEvalvideos(self):
        self.evalvideos = [eval for videoId, eval in list(self.videoToEval.items())]
