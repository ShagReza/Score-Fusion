# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:06:47 2020

@author: user
"""


#---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LdaSciKit
import scipy.io
#from sklearn.LpLda import LinearDiscriminantAnalysis as LpLdaSciKit
from sklearn.mixture import GaussianMixture as Gaussian
import itertools, random
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
#usage: alist = select(10,2)
def ListOfPositivePairs(size, pair_size):
    g =itertools.combinations(range(size),pair_size)
    alist = list(g)
    random.seed(4)
    random.shuffle(alist)
    return alist

def ListOfNegativePairs(list1, list2):
    alist = list(itertools.product(list1, list2))
    random.seed(4)
    random.shuffle(alist)
    return alist
#---------------------------------------------------------------------------
    




def ExtractFusionTrails(x_train,label_train,dim_lda):
    TrailDim=2*dim_lda
    #------------------------------------
    #positive trails:
    vectors=x_train
    labels=label_train
    unique_labels = np.unique(label_train)
    PositivePairs=np.empty((1,TrailDim))
    Pairs=np.empty((1,TrailDim))
    for label in unique_labels:          
        vecs = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
        print(label, len(vecs))     
        a=len(vecs)
        alist = ListOfPositivePairs(a,2)
        
        if len(alist)>0:
            L=25
            if len(alist)<L:
                L=len(alist)
            for x in range(L):
                A1=vecs[alist[x][0]]
                A2=vecs[alist[x][1]]
                Pairs=np.append(A1,A2)
                Pairs = np.reshape(Pairs, (1,TrailDim))
                PositivePairs=np.append(PositivePairs,Pairs,axis=0)
           
    PositivePairs=np.delete(PositivePairs,0,0)
    FusionTrails_positive=PositivePairs
    #np.save('FusionTrails_positive',FusionTrails_positive)
    #PositivePairs=np.load('PositivePairs.npy')
    #-----------------------------------
    # Negative trails:
    NegativePairs=np.empty((1,TrailDim))
    Pairs=np.empty((1,TrailDim))
    for label in unique_labels:   
        print(label)            
        vecs = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
        vecs_not = [vectors[i] for i in range(len(vectors)) if labels[i] != label]
        
        #alist = ListOfNegativePairs( list(range(len(vecs))) , list(range(len(vecs_not))) #time consuming
        
        A=list(range(len(vecs_not)))
        random.seed(4)
        random.shuffle(A)
        AA=A[0:100]
        alist = ListOfNegativePairs( list(range(len(vecs))) , AA )
        
        if len(alist)>0:
            L=20
            if len(alist)<L:
                L=len(alist)
            for x in range(L):
                A1=vecs[alist[x][0]]
                A2=vecs_not[alist[x][1]]
                Pairs=np.append(A1,A2)
                Pairs = np.reshape(Pairs, (1,TrailDim))
                NegativePairs=np.append(NegativePairs,Pairs,axis=0)
    NegativePairs=np.delete(NegativePairs,0,0)
    FusionTrails_negative=NegativePairs
    #np.save('FusionTrails_negative',FusionTrails_negative)
    #NegativePairs=np.load('NegativePairs.npy')
    
    return FusionTrails_negative , FusionTrails_positive