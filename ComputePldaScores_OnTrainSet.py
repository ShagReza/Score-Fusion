# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 08:48:43 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:30:31 2020

@author: user
"""
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()




def ComputePldaScores_OnTrainSet(FusionTrails_positive,FusionTrails_negative):

    
    x_test=FusionTrails_positive
    NumTest=len(x_test) 
    scores=[]
    for i in range(NumTest):
        print('positive trail',i)
        A1=x_test[i][0:150]
        A2=x_test[i][150:300]
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        s= eng.score_gplda_trials(A1matlab,A2matlab,nargout=1)
        scores.append(s)
    scores_Plda_positive=scores     
    
   
    x_test=FusionTrails_negative
    NumTest=len(x_test) 
    scores=[]
    for i in range(NumTest):
        print('negative trail',i)
        A1=x_test[i][0:150]
        A2=x_test[i][150:300]
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        s = eng.score_gplda_trials(A1matlab,A2matlab,nargout=1)
        scores.append(s)
    scores_Plda_negative=scores   
    
    eng.quit()
    return scores_Plda_positive,scores_Plda_negative