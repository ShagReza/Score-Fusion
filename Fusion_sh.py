# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:36:23 2020
Fusion
Shaghayegh Reza
"""

#----------------------------------------------------------------------
# step1: preprocessing
# step2: Extract Fusion trails 
# step3: train and run each method and compute scores for fusion trails 
# step4: run matlab code in python to compute fusion weights
# step5: run each method to compute test scores for each method
# step6: compute fused score for test and compute EER
#----------------------------------------------------------------------



#----------------------------------------------------------------------
import numpy as np
import scipy.io
#----------------------------------------------------------------------


#==============================================================================================
            # step1: preprocessing
from Preprocessings import Preprocessings
from ComputeCosineScore_OnTrainSet import cosinedis

dim_lda=150
x_train , x_test, label_train=Preprocessings(dim_lda)
#--------------------------------------
            # step2: Extract Fusion trails 
from ExtractFusionTrails import ExtractFusionTrails 
FusionTrails_negative , FusionTrails_positive=ExtractFusionTrails(x_train,label_train,dim_lda)
np.save('FusionTrails_negative',FusionTrails_negative)
np.save('FusionTrails_positive',FusionTrails_positive)
FusionTrails_positive=np.load('FusionTrails_positive.npy')
FusionTrails_negative=np.load('FusionTrails_negative.npy')
scipy.io.savemat('FusionData.mat', 
                 mdict={'FusionTrails_positive':FusionTrails_positive, 
                        'FusionTrails_negative':FusionTrails_negative}) 
#==============================================================================================
            # step3: methods
# cosine score
from ComputeCosineScore_OnTrainSet import ComputeCosineScore_OnTrainSet
scores_cosine_positive,scores_cosine_negative=ComputeCosineScore_OnTrainSet(FusionTrails_positive,FusionTrails_negative)
np.save('scores_cosine_positive',scores_cosine_positive)
np.save('scores_cosine_negative',scores_cosine_negative)

# plda score
from ComputePldaScores_OnTrainSet import ComputePldaScores_OnTrainSet
scores_Plda_positive,scores_Plda_negative=ComputePldaScores_OnTrainSet(FusionTrails_positive,FusionTrails_negative)
np.save('scores_Plda_positive',scores_Plda_positive)
np.save('scores_Plda_negative',scores_Plda_negative)


# GaussinaBackend score
from ComputeGuassianScores_OnTrainSet import ComputeGuassianScores_OnTrainSet
scores_guassian_positive,scores_guassian_negative=ComputeGuassianScores_OnTrainSet(FusionTrails_positive,FusionTrails_negative)
np.save('scores_guassian_positive',scores_guassian_positive)
np.save('scores_guassian_negative',scores_guassian_negative)

# moPLDA score
dim_lda=250
x_train250 , x_test250, label_train=Preprocessings(dim_lda)
from ExtractFusionTrails import ExtractFusionTrails 
FusionTrails_negative250 , FusionTrails_positive250=ExtractFusionTrails(x_train250,label_train,dim_lda)
np.save('FusionTrails_negative250',FusionTrails_negative250)
np.save('FusionTrails_positive250',FusionTrails_positive250)
FusionTrails_positive250=np.load('FusionTrails_positive250.npy')
FusionTrails_negative250=np.load('FusionTrails_negative250.npy')

from ComputeMopldaScores_OnTrainSet import ComputeMopldaScores_OnTrainSet
scores_Moplda_positive,scores_Moplda_negative=ComputeMopldaScores_OnTrainSet(FusionTrails_positive250,FusionTrails_negative250)
np.save('scores_Moplda_positive',scores_Moplda_positive)
np.save('scores_Moplda_negative',scores_Moplda_negative)
#==============================================================================================
            # step4: Fusion weights
# run matlab code  !!!!!!!!!!!!!!!!!!
# input: scores  output: weight with llr method!!!!!!!!!!!!!!!!!!
# apply mapstd 
import matlab.engine
eng = matlab.engine.start_matlab()

scores_Moplda_positive.mean(axis=0) #axis???? 
scores_Moplda_positive.std(axis=0)  
scores_Moplda_negative.mean(axis=0)  
scores_Moplda_negative.std(axis=0)   
#
scores_guassian_positive.mean(axis=0)  
scores_guassian_positive.std(axis=0)  
scores_guassian_negative.mean(axis=0)  
scores_guassian_negative.std(axis=0)  
#
scores_Plda_positive.mean(axis=0)  
scores_Plda_positive.std(axis=0)  
scores_Plda_negative.mean(axis=0)  
scores_Plda_negative.std(axis=0)  
#
scores_cosine_positive.mean(axis=0)  
scores_cosine_positive.std(axis=0)  
scores_cosine_negative.mean(axis=0)  
scores_cosine_negative.std(axis=0)  
#   
TargetScores=[] #eslah kon append ro !!!!!!!   
TargetScores.append(TargetScores,scores_cosine_positive,axis=1)#axis???
TargetScores.append(TargetScores,scores_Plda_positive,axis=1)
TargetScores.append(TargetScores,scores_guassian_positive,axis=1)
TargetScores.append(TargetScores,scores_Moplda_positive,axis=1)
#
NonTargetScores=[]    
NonTargetScores=np.append(NonTargetScores,scores_cosine_negative,axis=1)#axis???
NonTargetScores=np.append(NonTargetScores,scores_Plda_negative,axis=1)
NonTargetScores=np.append(NonTargetScores,scores_guassian_negative,axis=1)
NonTargetScores=np.append(NonTargetScores,scores_Moplda_negative,axis=1)
#
TargetScores_matlab = matlab.double(TargetScores.tolist())
NonTargetScores_matlab = matlab.double(TNonTargetScores.tolist())
Weights= eng.score_moplda_trials(TargetScores_matlab,NonTargetScores_matlab,nargout=1)   
np.save('Weights',Weights)   
#==============================================================================================
            # step5: test results  
NumTest=900
m=0
A=np.arange(1, NumTest+1, 1) 
score_cosine=[]

InvCovP=np.load('InvCovP')  
InvCovN=np.load('InvCovN')  
mean_P=np.load('mean_P')  
mean_N=np.load('mean_N')  
           
for i,j in enumerate(A):
    B=np.arange(j+1,NumTest+1,1)
    for k,l in enumerate(A) :
        A1=x_test[j]
        A2=x_test[l]
        # cosine score
        score_cosine.append(cosinedis(A1,A2))
        # PLDA score
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        score_plda= eng.score_gplda_trials(A1matlab,A2matlab,nargout=1)
        # Guassian score        
        Pairs=np.append(A1,A2)
        Pairs = np.reshape(Pairs, (1,300))
        score_guassian=-1*(Pairs-mean_P).dot(InvCovP).dot(np.transpose(Pairs-mean_P))+(Pairs-mean_N).dot(InvCovN).dot(np.transpose(Pairs-mean_N))
        # Moplda score
        A1=x_test250[j]
        A2=x_test250[l]
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        score_moplda= eng.score_moplda_trials(A1matlab,A2matlab,nargout=1)
        
ScrTotal=[]
ScrTotal.append(score_cosine,axis=0)
ScrTotal.append(score_plda,axis=0)
ScrTotal.append(score_guassian,axis=0)
ScrTotal.append(score_moplda,axis=0)
np.save('ScrTotla',ScrTotal)  
#==============================================================================================
            # step6: compute EER
FusedScore=np.transpose(Weights).dot(ScrTotal)
ypred=FusedScore
scipy.io.savemat('ypred.mat',ypred)
#==============================================================================================            
      
            