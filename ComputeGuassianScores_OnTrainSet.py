# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:30:31 2020

@author: user
"""
from sklearn.mixture import GaussianMixture as Gaussian
import numpy as np

def ComputeGuassianScores_OnTrainSet(FusionTrails_positive,FusionTrails_negative):

    NegativePairs=np.load('NegativePairs.npy')
    PositivePairs=np.load('PositivePairs.npy')
    
    #8: Train Gaussian models
    GaussainPositive=Gaussian(n_components=1,covariance_type='full')
    GaussainNegative=Gaussian(n_components=1,covariance_type='full')
    GaussainPositive.fit(PositivePairs)
    GaussainNegative.fit(NegativePairs)
    cov_P=GaussainPositive.covariances_
    mean_P=GaussainPositive.means_
    cov_N=GaussainNegative.covariances_
    mean_N=GaussainNegative.means_
    

    cov_N = np.reshape(cov_N, (300,300))
    cov_P = np.reshape(cov_P, (300,300))
    InvCovP=np.linalg.inv(cov_P)
    InvCovN=np.linalg.inv(cov_N)
    #-------------------------------------------
    #9: Compute Scores
    x_test=FusionTrails_positive
    NumTest=len(x_test) 
    scores=[]
    for i in range(NumTest):
        A1=x_test[i][0:150]
        A2=x_test[i][150:300]
        Pairs=np.append(A1,A2)
        Pairs = np.reshape(Pairs, (1,300))
        s=-1*(Pairs-mean_P).dot(InvCovP).dot(np.transpose(Pairs-mean_P))+(Pairs-mean_N).dot(InvCovN).dot(np.transpose(Pairs-mean_N))
        scores.append(s[0][0])
    scores_guassian_positive=scores     
    
    x_test=FusionTrails_negative
    NumTest=len(x_test)    
    scores=[]
    for i in range(NumTest):
        A1=x_test[i][0:150]
        A2=x_test[i][150:300]
        Pairs=np.append(A1,A2)
        Pairs = np.reshape(Pairs, (1,300))
        s=-1*(Pairs-mean_P).dot(InvCovP).dot(np.transpose(Pairs-mean_P))+(Pairs-mean_N).dot(InvCovN).dot(np.transpose(Pairs-mean_N))
        scores.append(s[0][0])
    scores_guassian_negative=scores  
    
    np.save('InvCovP',InvCovP)  
    np.save('InvCovN',InvCovN)  
    np.save('mean_P',mean_P)  
    np.save('mean_N',mean_N)  
    return scores_guassian_positive,scores_guassian_negative