
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()




def ComputeMopldaScores_OnTrainSet(FusionTrails_positive,FusionTrails_negative):

    
    x_test=FusionTrails_positive
    NumTest=len(x_test) 
    scores=[]
    for i in range(NumTest):
        print('positive trail',i)
        A1=x_test[i][0:250]
        A2=x_test[i][250:500]
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        s= eng.score_moplda_trials(A1matlab,A2matlab,nargout=1)
        scores.append(s)
    scores_Moplda_positive=scores     
    
   
    x_test=FusionTrails_negative
    NumTest=len(x_test) 
    scores=[]
    for i in range(NumTest):
        print('negative trail',i)
        A1=x_test[i][0:250]
        A2=x_test[i][250:500]
        A1matlab = matlab.double(A1.tolist())
        A2matlab = matlab.double(A2.tolist())
        s = eng.score_moplda_trials(A1matlab,A2matlab,nargout=1)
        scores.append(s)
    scores_Moplda_negative=scores   
    
    eng.quit()
    return scores_Moplda_positive,scores_Moplda_negative