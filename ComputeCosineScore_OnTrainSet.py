# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:14:04 2020

@author: user
"""
import numpy as np

def cosinedis(v1,v2):
  dot_product = np.dot(v1, v2)
  norm_a = np.linalg.norm(v1)
  norm_b = np.linalg.norm(v2)
  return dot_product / (norm_a * norm_b)



def ComputeCosineScore_OnTrainSet(FusionTrails_positive,FusionTrails_negative):
    y_pred=[]
    data=FusionTrails_positive
    for i in range(len(data)):
        file1 =data[i][0:150]
        file2= data[i][150:300]
        res = cosinedis(file1,file2)
        y_pred.append(res)    
    scores_cosine_positive=y_pred
    
    y_pred=[]
    data=FusionTrails_negative
    for i in range(len(data)):
        file1 =data[i][0:150]
        file2= data[i][150:300]
        res = cosinedis(file1,file2)
        y_pred.append(res)    
    scores_cosine_negative=y_pred
    
    return scores_cosine_positive,scores_cosine_negative