#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2
import numpy as np
from pyemd import emd
from PIL import Image
import skimage.color
from collections import defaultdict
import pandas as pd


def image_colorfulness(image):
    '''calculate colorfulness from picture
    https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/'''
    # split the image into its respective RGB components
    B, G, R = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def image_colorfulness_emd(im_name):

    im = Image.open(im_name)
    pix = im.load()

    h1 = [1.0/64] * 64
    h2 = [0.0] * 64
    hist1 = np.array(h1)

    hist = cv2.calcHist([img], [0, 1, 2], None, [4,4,4], [0, 256, 0, 256, 0, 256])

    w,h = im.size

    for x in range(w):
        for y in range(h):
            cbin = int(pix[x,y][0]/64*16 + pix[x,y][1]/64*4 + pix[x,y][2]/64)
            h2[cbin-1]+=1
    hist2 = np.array(h2)/w/h

    # compute center of cubes

    c = np.zeros((64,3))
    for i in range(64):
        b = (i%4) * 64 + 32
        g = (i%16/4) * 64 + 32
        r = (i/16) * 64 + 32
        c[i]=(r,g,b)

    c_luv = skimage.color.rgb2luv(c.reshape(8,8,3)).reshape(64,3)

    d = np.zeros((64,64))

    for x in range(64):
        d[x,x]=0
        for y in xrange(x):
            dist = np.sqrt( np.square(c_luv[x,0]-c_luv[y,0]) + 
                      np.square(c_luv[x,1]-c_luv[y,1]) + 
                      np.square(c_luv[x,2]-c_luv[y,2]))
            d[x,y] = dist
            d[y,x] = dist


    colorfullness = emd(hist1, hist2, d)

    return colorfullness


def compute_histogram(src, h_bins = 30, s_bins = 32, scale = 10):
    '''calculate histogram from picture'''
    #create images
    hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
    hplane = cv.CreateImage(cv.GetSize(src), 8, 1)
    splane = cv.CreateImage(cv.GetSize(src), 8, 1)
    vplane = cv.CreateImage(cv.GetSize(src), 8, 1)

    planes = [hplane, splane]
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)
    cv.CvtPixToPlane(hsv, hplane, splane, vplane, None)

    #compute histogram
    hist = cv.CreateHist((h_bins, s_bins), cv.CV_HIST_ARRAY,
            ranges = ((0, 180),(0, 255)), uniform = True)
    cv.CalcHist(planes, hist)      #compute histogram
    cv.NormalizeHist(hist, 1.0)    #normalize histo

    return hist


def compute_signatures(hist1, hist2, h_bins = 30, s_bins = 32):
    '''
    demos how to convert 2 histograms into 2 signature
    '''
    num_rows = h_bins * s_bins
    sig1 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    sig2 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    #fill signatures
    #TODO: for production optimize this, use Numpy
    for h in range(0, h_bins):
        for s in range(0, s_bins):
            bin_val = cv.QueryHistValue_2D(hist1, h, s)
            cv.Set2D(sig1, h*s_bins + s, 0, bin_val) #bin value
            cv.Set2D(sig1, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig1, h*s_bins + s, 2, s) #coord2
            #signature.2
            bin_val2 = cv.QueryHistValue_2D(hist2, h, s)
            cv.Set2D(sig2, h*s_bins + s, 0, bin_val2) #bin value
            cv.Set2D(sig2, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig2, h*s_bins + s, 2, s) #coord2

    return (sig1, sig2)
    

def compute_emd(src1, src2, h_bins, s_bins, scale):
    hist1  = compute_histogram(src1, h_bins, s_bins, scale)
    hist2  = compute_histogram(src2, h_bins, s_bins, scale)
    sig1, sig2 = compute_signatures(hist1, hist2)
    emd = cv.CalcEMD2(sig1, sig2, cv.CV_DIST_L2)
    return emd


if __name__ == "__main__":

    ## TODO: get unsupervised metrics: ideas in The Interestingness of Images
    # vivid color/ color harmony, good lightning, saliency
    # alse check with partners data if there is something in common with the predicted features

    viz = True
    im_path = './prediction_images/'

   
    feat_dict = defaultdict(list)
    im_list = os.listdir(im_path)
    for im_name in im_list:

      # Load image
      im = cv2.imread(im_path+im_name)
      feat_dict['ImageFile'].append(im_name)

      # colorfulness: as the Earth Mover distance (in the LUV color space) of the color histogram of an image HI to a uniform color histogram Huni. 
      # A uniform color histogram is the most colorful possible, thus the smaller the distance, the more colorful the image
      # im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2Luv)
      #hist_i = cv2.calcHist([im_luv])
      #compute_emd(im_luv, src2, h_bins, s_bins, scale)

      # version 2
      colorfulness = image_colorfulness(im)
      feat_dict['colorfulness'].append(colorfulness)

      if viz:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()
        print(colorfulness)
    
    df = pd.DataFrame(data=feat_dict)


      


            
