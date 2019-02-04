import time
t = time.time()
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import matplotlib
import os
import pandas as pd
import natsort
os.chdir('D:\\Algorithms\\NeuroScience\\Updated SpaceCat\\Data\\True\\Mutilation')
files = os.listdir()
files = natsort.natsorted(files)
Data = np.ndarray(())
Trial_No = np.zeros((30))
n = 0
for file in range(0,len(files),4):
    matl = sio.loadmat(files[file])['outmat']
    if len(Data.shape)==0:
        Data = matl
    elif len(matl.shape)==2: 
        matl = np.reshape(matl, (129,1501,1))
        Data =np.append(Data,matl,axis=2)
    else:
        Data =np.append(Data,matl,axis=2)
        print(file)
    Trial_No[n] = matl.shape[2]
    n += 1
#np.save('True_Mutilation_LeastBlury.npy',Data)
np.save('TrailNo_True_Mutilation_LeastBlury.npy',Trial_No)
elapsed = time.time() - t
print('The Calculation time in Second is:', elapsed)