# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:56:15 2020

@author: user
"""
#---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LdaSciKit
import scipy.io
#from sklearn.LpLda import LinearDiscriminantAnalysis as LpLdaSciKit
from sklearn.mixture import GaussianMixture as Gaussian
#---------------------------------------------------------------------------






def Preprocessings(dim_lda):  
    #---------------------------------------------------------------------------
    # 1: load train data
    filename='xvectors.npz'  
    with np.load(filename , allow_pickle=True) as da:
        train_name = da['spk_name']
        x_train= da['features']    
    label_train = pd.factorize(train_name)[0]
    Num_spks_train=np.max(label_train)
    #load test data
    filename='test_kermanshah_xvec.npz'
    with np.load(filename , allow_pickle=True) as da:
        test_name = da['data_name']
        x_test = da['features']
    #--------------------------------------------------
    # 2: compute mean (save it)
    m = np.mean(x_train, axis=0)
    np.save('m',m)
    #--------------------------------------------------
    # 3: compute whitening transform (save it)
    S = np.cov(x_train, rowvar=0)
    D, V = np.linalg.eig(S)
    W = (1/np.sqrt(D) * V).transpose().astype('float32')
    np.save('W',W)
    #--------------------------------------------------
    # 4-1: Apply mean 
    x_train2= x_train - m
    x_test2= x_test - m
    # 4-2: Apply mean and whitening
    x_train3= np.dot(x_train2, W.transpose())
    x_test3= np.dot(x_test2, W.transpose())
    #--------------------------------------------------
    #choose between x_train2 and x_train3!!!!!!!!!!!!!!
    x_train=x_train3 #!!!!!!! without whitening in moplda
    x_test=x_test3 #!!!!!!! without whitening in moplda
    #--------------------------------------------------
    # 5: extract LDA model (save it)
    #dim_lda=150 #250 in moplda!!!!!!!!!!!
    lda = LdaSciKit(n_components=dim_lda,solver='eigen')
    lda.fit(x_train, label_train)
    # 6: apply LDA 
    x_train_lda=lda.transform(x_train)
    x_test_lda=lda.transform(x_test)
    x_train=x_train_lda
    x_test=x_test_lda
    #length normalization (map into unit sphere):
    x_train /= np.sqrt(np.sum(x_train ** 2, axis=1))[:, np.newaxis]  
    x_test /= np.sqrt(np.sum(x_test ** 2, axis=1))[:, np.newaxis] 
    
    return x_train , x_test, label_train
