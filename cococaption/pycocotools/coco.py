from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
__version__ = '1.0.1'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
# from skimage.draw import polygon
import copy
import ipdb

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.videoToAnns = {}
        self.videos = []
        if not annotation_file == None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            if 'type' not in dataset:
                dataset['type']='caption'
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        videoToAnns = {ann['video_id']: [] for ann in self.dataset['annotations']}
        #ipdb.set_trace()
        #anns =      {ann['sen_id']:       [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            videoToAnns[ann['video_id']] += [ann]
            #anns[ann['sen_id']] = ann

        # videos = {vi['video_id']: {} for vi in self.dataset['videos']}
        # for vi in self.dataset['videos']:
        #     videos[vi['video_id']] = vi

        videos = {vi['video_id']: {} for vi in self.dataset['annotations']}

        print('index created!')

        # create class members
        #self.anns = anns
        self.videoToAnns = videoToAnns
        self.videos = videos


    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in list(self.datset['info'].items()):
            print('%s: %s'%(key, value))

    
    def getVideoIds(self):
        return list(self.videos.keys())
        
    
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        #res.dataset['videos'] = [vi for vi in self.dataset['videos']]
        #res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        #res.dataset['type'] = copy.deepcopy(self.dataset['type'])
        

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        #anns    = json.load(open(resFile))
        anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsVideoIds = [ann['video_id'] for ann in anns]
        # print(len(set(annsVideoIds)))
        # print(set(annsVideoIds))
        # print(set(self.getVideoIds()))
        # print(len(set(self.getVideoIds())))
        # import ipdb
        # ipdb.set_trace()
        assert set(annsVideoIds) == (set(annsVideoIds) & set(self.getVideoIds())), \
               'Results do not correspond to current coco set'
    
        #videoIds = set([vi['video_id'] for vi in res.dataset['videos']]) & set([ann['video_id'] for ann in anns])
        #res.dataset['videos'] = [vi for vi in res.dataset['videos'] if vi['video_id'] in videoIds]
        # for id, ann in enumerate(anns):
        #     ann['sen_id'] = id
        
        print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
