# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scorer.cider_scorer import CiderScorer

class Cider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, n=4, sigma=6.0, document_frequency=None, ref_len=None):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

        self.cider_scorer = CiderScorer(n=self._n, sigma=self._sigma, document_frequency=document_frequency, ref_len=ref_len)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        # clear all the previous hypos and refs
        self.cider_scorer.clear()
        for i, hypo in enumerate(res):
            ref = gts[i]

            # Sanity check.
            #assert(type(hypo) is list)
            #assert(len(hypo) == 1)
            # assert(type(ref) is list)
            # assert(len(ref) > 0)

      
            self.cider_scorer += (hypo, ref)

        (score, scores) = self.cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"    