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


def norm_data(x, minim, maxim):
  return (x-minim)/(maxim-minim)


def binarize(x, th=0.6): # th=0.2
  return 1 if float(x) > th else 0


def binarize_data(data, th):
  """binarize data given a threshold.
  """
  return {
    key: binarize(val, th) for key, val in data.items() 
  }


def create_method_pred_df(data, method = '1'):
  """creates a pandas dataframe with the predictions given the method {'1','2','3','4'}.
  """
  df_pred = pd.DataFrame.from_dict({i: data[i][method] for i in data.keys() for j in data[i].keys()}, orient='index')
  df_pred.reset_index(inplace=True)
  df_pred = df_pred.rename(columns = {'index':'ImageFile'}).sort_values(by=['ImageFile'])
  return df_pred 


def evaluate_spearman_corr(gt, pred):
  imgs_to_eval = list(set(gt.ImageFile) & set(pred.ImageFile))
  gt = gt[gt['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])
  pred = pred[pred['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])

  for col in pred.columns[1:]:
    attr_gt = gt.loc[:,'TotalScore']
    attr_pred = pred.loc[:,col]
    rho,pval = spr(attr_gt,attr_pred)
    print("{}: rho: {} at p value: {}".format(col, rho, pval))    

  # rho,pval = spr(gt.loc[:,'TotalScore'], pred.loc[:,feature])
  return


def evaluate_parnaca(df_parnaca_gt, df_parnaca_pred):
  """Spearman’s ρ rank correlation statistic is used.
  """
  imgs_to_eval = list(set(df_parnaca_gt.ImageFile) & set(df_parnaca_pred.ImageFile))
  gt = df_parnaca_gt[df_parnaca_gt['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])
  pred = df_parnaca_pred[df_parnaca_pred['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])
  for col in gt.columns[1:]:
    attr_gt = gt.loc[:,col]
    attr_pred = pred.loc[:,col]
    rho,pval = spr(attr_gt,attr_pred)
    #print("-----------------", col)
    #print(" rho: {} at p value: {}".format(rho,pval))    
  return rho


def plot_corr(gt, pred, name):
  imgs_to_eval = list(set(gt.ImageFile) & set(pred.ImageFile))
  gt = gt[gt['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])
  pred = pred[pred['ImageFile'].isin(imgs_to_eval)].sort_values(by=['ImageFile'])

  for col in pred.columns[1:]:
    attr_gt = gt.loc[:,'TotalScore']
    attr_pred = pred.loc[:,col]
    df = pd.DataFrame({"AestheticScore": attr_gt, col: attr_pred})
    sns_scatter_plot = sns.jointplot(x="AestheticScore", y=col, data=df, kind="reg")
    rho, pval = spr(attr_gt,attr_pred)
    #sns_scatter_plot.ax_joint.text(0.2,0.2,"{:.4f}".format(rho))
    sns_scatter_plot.savefig("./eval/" + name + "_" + col + "_" + "{:.4f}".format(rho) + ".png")
    plt.close()
    #plt.show()


if __name__ == "__main__":

    viz = False
    dataset = "AADB" 
    
    if dataset == "AADB":
      gt_path = Path("../Datasets/AADB_ground_truth/")
      label_name = Path(gt_path, 'labels.csv')
      df_parnaca_gt = pd.read_csv(label_name)
      df_parnaca_gt.ImageFile.tolist()
      # label_name = Path(gt_path, 'labels.json')  
      # with open(label_name) as json_file:
      #   parnaca_labels = json.load(json_file)
    
    # Read predictions
    with open('prediction_results.json') as json_file:
        data = json.load(json_file)

        # Global evaluation: all the images in the folder
        df_mlsp_pred = create_method_pred_df(data, method = '1')      
        df_deepface_pred = create_method_pred_df(data, method = '2')
        df_hecate_pred = create_method_pred_df(data, method = '3')
        df_parnaca_pred = create_method_pred_df(data, method = '4')
        df = pd.merge(pd.merge(pd.merge(df_mlsp_pred, df_deepface_pred), df_hecate_pred), df_parnaca_pred)

        corr_parnaca_score = evaluate_parnaca(df_parnaca_gt, df_parnaca_pred)

        # corr between colorharmony and totalscore
        evaluate_spearman_corr(df_parnaca_gt, df_parnaca_pred)
        evaluate_spearman_corr(df_parnaca_gt, df_parnaca_gt)
        evaluate_spearman_corr(df_parnaca_gt, df_hecate_pred)

        plot_corr(df_parnaca_gt, df_parnaca_gt, name = "parnaca_gt")
        plot_corr(df_parnaca_gt, df_hecate_pred, name = "hecate")
       
        # Individual evaluation: per image 
        for im_name in data:
          im_info = data[im_name]

          # methods
          mlsp = im_info['1']
          deepface = im_info['2']
          hecate = im_info['3']
          parnaca_predictions = im_info['4']

          # 1- Mlsp Aesthetic Score Predictor: aesthetic score [0,10]
          aesthetic_score = norm_data(mlsp['aesthetic score'], 0, 10) 

          # 2- Deepface Emotion Predictor: none, surprised, happy, sad, ...
          facial_emotion = deepface['facial emotion']

          # 3- Hecate Image Metrics Predictor
          Asymmetry = hecate['Asymmetry'] # 0-1
          Brightness = hecate['Brightness'] # 0-1
          ContrastBalance = hecate['ContrastBalance']
          Entropy = hecate['Entropy']
          ExposureBalance = hecate['ExposureBalance']
          JpegQuality = hecate['JpegQuality']
          RMSContrast = hecate['RMSContrast']
          Sharpness = hecate['Sharpness'] # 0-1
          Uniformity = hecate['Uniformity']

          # 4- Parnaca Image Metrics Predictor: individual metrics [-1,1] / TotalScore [0,1]
          # They do: negative [-1,-0.2], null [-0.2,0.2], positive [0.2,1]. Try to find a better threshold
          parnaca_gt = df_parnaca_gt[df_parnaca_gt['ImageFile']==im_name].to_dict('records')[0]
          parnaca_accuracy = evaluate_parnaca(parnaca_predictions, parnaca_gt)                 

          if abs(aesthetic_score - float(parnaca_predictions["TotalScore"])) > 0.05:
            print("score discrepancy: ", abs(aesthetic_score - float(parnaca_predictions["TotalScore"])))

          if viz:
            img = mpimg.imread(Path('./prediction_images/', im_name))
            plt.imshow(img)
            plt.show()


          
