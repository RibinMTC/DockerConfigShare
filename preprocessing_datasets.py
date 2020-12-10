#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import pandas as pd
from sys import argv
import os
import shutil


def preprocess_parnaca_gt(gt_path, phase='TestNew'):
  """Create gt dictionary with the following structure:

    file_name1:
      feature 1: score
      feature 2: score
      ...
    file_name2:
      ...
  """
  # phase = ['Train', 'TestNew', 'Validation']
  parnaca_keys = ['TotalScore', 'BalancingElement', 'ColorHarmony', 'InterestingContent', 'DoF', 'GoodLighting', 'MotionBlur', 'ObjectEmphasis', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']
  file_prefix = 'imgListTestNewRegression_'
  gt_dict = {}

  for feat_key in parnaca_keys:
    
    # correct typos in their file names...
    feat_key_ = feat_key
    if feat_key == 'BalancingElement':
      feat_key_ = 'BalacingElements'
    elif feat_key == 'GoodLighting':
      feat_key_ = 'Light'
    elif feat_key == 'InterestingContent':
      feat_key_ = 'Content'
    elif feat_key == 'ObjectEmphasis':
      feat_key_ = 'Object'
    elif feat_key == 'TotalScore':
      feat_key_ = 'score'

    # read file
    file_path = Path(gt_path, file_prefix + feat_key_ + '.txt')
    f = open(file_path, "r")

    # create and save gt dir
    for line in f:
      im_key, value = line.split()
      if im_key not in gt_dict:
        gt_dict[im_key] = {}
      gt_dict[im_key][feat_key] = value

  json_name = Path(gt_path, 'labels.json')  
  with open(json_name, 'w') as json_file:
    json.dump(gt_dict, json_file)


def preprocess_parnaca_gt2(gt_path, phase='TestNew'):
  # it will generate the file labels.csv containing all the attributes and scores for all the images 

  file_prefix = 'imgListTestNewRegression_'

  attr = ["BalacingElements","ColorHarmony","Content","DoF","Light","MotionBlur","Object",
          "Repetition","RuleOfThirds","Symmetry","VividColor","score"]

  testFiles = []
  for item in attr:
      file_path = Path(gt_path, file_prefix + str(item) + '.txt')
      testFiles.append(file_path)
  
  df = pd.read_csv(testFiles[0],delimiter = ' ',header=None)
  df.columns = ['ImageFile', 'BalancingElement']
  for i in range(1,len(testFiles)):
      new_df = pd.read_csv(testFiles[i],delimiter = ' ',header=None)
      
      attr_name = attr[i]
      if attr_name == 'Light':
        attr_name = 'GoodLighting'
      elif attr_name == 'Content':
        attr_name = 'InterestingContent'
      elif attr_name == 'Object':
        attr_name = 'ObjectEmphasis'
      elif attr_name == 'score':
        attr_name = 'TotalScore'
      new_df.columns = ['ImageFile', attr_name]

      df = pd.merge(df,new_df)

  df_name = Path(gt_path, 'labels.csv') 
  df.to_csv(df_name,index=False)


def separate_gt_phases(gt_path, im_path, phase='test'):

  label_name = Path(gt_path, 'labels.csv')
  df_parnaca_gt = pd.read_csv(label_name)
  im_test_list = df_parnaca_gt.ImageFile.tolist()
  for im in im_test_list:
    source = Path(im_path, im)
    destination = Path(im_path, phase, im)
    if os.path.exists(source):
      shutil.move(source, destination)


if __name__ == "__main__":


  # AADB dataset
  gt_path = Path("../Datasets/AADB_ground_truth/")
  im_path = Path("../Datasets/AADB_images/")
  #preprocess_parnaca_gt(gt_path)
  #preprocess_parnaca_gt2(gt_path)   
  separate_gt_phases(gt_path, im_path, phase='test')

