# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:13:08 2020

@author: user
"""

import numpy  as np
import matlab
import matlab.engine
eng =  matlab.engine.start_matlab()

eng.cd()

Nn = 30
x= 250*np.ones((1,Nn)) 
y= 100*np.ones((1,Nn)) 
z = 32
xx = matlab.double(x.tolist())
yy = matlab.double(y.tolist())

Output = eng.simple_test(A2matlab,A1matlab,nargout=2)




scr = eng.score_gplda_trials(A2matlab,A1matlab,nargout=1)