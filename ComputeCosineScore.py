# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:43:57 2020

@author: user
"""

import numpy as np
import scipy.io
from utils import saveplot,load_data

def ComputeCosineScore(x_test):

    D2 = scipy.io.loadmat('test_ivecs.mat') 
    D2['Results']['ivectors'][0,0]=np.transpose(x_test)
    scipy.io.savemat('test_ivecs_filtered.mat', D2)

    
    l = [0 , 0.0001 , .001 , .01 , 0.1 , 0.2,0.3,0.4,0.5,0.75,1]     
    dic=load_data('test_ivecs_filtered.npz' )  
    scores_cosine=saveplot(dic , l , 'test-all.scp','ALL' , 'Ivector_raw' , 'Cosine' , 'log.txt','\t',1)
    return scores_cosine