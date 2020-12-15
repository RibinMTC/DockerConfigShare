#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import spearmanr as spr
import pandas as pd
import pdb
import seaborn as sns
import numpy as np


def draw(img, predictor, viz=False, save=False, direc="", name=""):
  """Visualize or save predicted bounding boxes"""
  plt.cla()
  plt.axis('off')
  plt.imshow(img)

  for pred in predictor:
      bbox = pred['box']
      if 'detectedEmotion' in pred.keys(): 
        emotion = pred['detectedEmotion']
      else:
        emotion = max(pred['emotions'], key=pred['emotions'].get)

      color = (np.random.rand(), np.random.rand(), np.random.rand())
      rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor=color, linewidth=2.5)
      plt.gca().add_patch(rect)
      plt.gca().text(bbox[0], bbox[1], '{:s}'.format(emotion), bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

  if viz:
    plt.show()
  
  if save:
    #plt.savefig(direc + '/%s_det_%i.jpg' % (name[:-4], int(100 * args.eval_min_conf)), bbox_inches='tight')
    plt.savefig(direc + name, bbox_inches='tight')


def create_method_pred_df(data, method = '1'):
  """creates a pandas dataframe with the predictions given the method {'1','2','3','4'}.
  """
  df_pred = pd.DataFrame.from_dict({i: data[i][method] for i in data.keys() for j in data[i].keys()}, orient='index')
  df_pred.reset_index(inplace=True)
  df_pred = df_pred.rename(columns = {'index':'ImageFile'}).sort_values(by=['ImageFile'])
  return df_pred 


if __name__ == "__main__":

    viz = False
    
    with open('prediction_results.json') as json_file:
        data = json.load(json_file)
  
    # -------- Individual evaluation: per image --------
    for im_name in data:
      img = mpimg.imread(Path('./prediction_images/', im_name))
      im_info = data[im_name]

      # methods
      paz = im_info['5']
      fer = im_info['6']

      if paz:
        draw(img, paz, save=True, direc="./emotions/", name='2'+im_name)
      
      if fer:
        draw(img, fer, save=True, direc="./emotions/", name=im_name)

      #pdb.set_trace()


      if viz:
        plt.imshow(img)
        plt.show()


          
