# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
# TOR-related DBs
from lib.datasets.gtea import gtea
from lib.datasets.gtea_gaze_plus import gtea_gaze_plus
from lib.datasets.tego import tego
from lib.datasets.tor_feedback import tor_feedback

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up GTEA
for split in ["train", "val", "test", "train-wholeBB", "val-wholeBB", "test-wholeBB"]:
  name = "gtea_{}".format(split)
  __sets[name] = (lambda split=split: gtea(split))

# Set up GTEA Gaze+
for split in ["train", "val", "test", "train-wholeBB", "val-wholeBB", "test-wholeBB"]:
  name = "gtea-gaze-plus_{}".format(split)
  __sets[name] = (lambda split=split: gtea_gaze_plus(split))

# Set up TEgO
for split in ["train", "val", "test", "train-wholeBB", "val-wholeBB", "test-wholeBB",\
              "train-blind", "train-sighted", "train-blind-wholeBB", "train-sighted-wholeBB"]:
  name = "tego_{}".format(split)
  __sets[name] = (lambda split=split: tego(split))

# Set up TOR feedback
for split in ["train", "val", "test", "train-wholeBB", "val-wholeBB", "test-wholeBB"]:
  name = "tor-feedback_{}".format(split)
  __sets[name] = (lambda split=split: tor_feedback(split))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
