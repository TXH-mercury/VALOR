from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .spice.spice import Spice

class SpiceEval():
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.spice = Spice()
        self.tokenizer = PTBTokenizer()

    """
    The input have structure
    {'123': [{'image_id':123, 'caption': 'xxxxx'}, {'image_id':123, 'caption': 'yyy'}], ...}
    """
    def evaluate(self, gts, res):
        assert set(gts.keys()) == set(res.keys())
        imgIds = list(gts.keys())
        gts  = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================

        # =================================================
        # Compute scores
        # =================================================
        print('computing %s score...'%(self.spice.method()))
        score, scores = self.spice.compute_score(gts, res)
        print("%s: %0.3f"%("spice", score))
        self.eval['spice'] = score
        print(scores)
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId]["spice"] = score
        return self.eval['spice'], self.imgToEval
        # self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]


class COCOEvalCapSpice:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

        self.Spice = Spice()

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (self.Spice, "SPICE")
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
                    self.setImgToEvalImgs(scs, list(gts.keys()), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, list(gts.keys()), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]
