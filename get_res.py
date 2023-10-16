

#----------------------------------------------------
from utils import compute_eer,findFR
from sklearn.metrics import roc_curve as rc
import scipy.io
#----------------------------------------------------



#----------------------------------------------------
y = scipy.io.loadmat('y.mat')
y=y['y']
y=y.tolist()[0]

ypred = scipy.io.loadmat('ypred.mat')
ypred=ypred['ypred']
ypred=ypred.tolist()[0]
#----------------------------------------------------


#----------------------------------------------------
fpr, tpr, threshold = rc(y, ypred, pos_label=1)
eer , th =  compute_eer(fpr,tpr,threshold) 
print(eer*100)
fr = findFR(0 , fpr, tpr) 
print(fr)
#----------------------------------------------------