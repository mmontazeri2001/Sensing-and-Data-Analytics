import time
t = time.time()
import numpy as np
import os
from scipy.spatial import distance
os.chdir('D:\\Algorithms\\NeuroScience\\Updated SpaceCat\\Data\\True')
"""Set the Dataset  """
#Data = np.load('True_Neutral.npy')
Data = np.load('True_Mutilation.npy')
Data_Size = Data.shape[2]
Eig_Value = np.array([])
for i in range(Data_Size):
#for i in range(3):
    Sel_sensors =[51, 52, 53, 58, 59, 60, 61, 65, 66, 70, 71, 74, 75, 76, 77, 78, 83, 84, 85, 90, 91]
    #Sel_sensors = range(129)
    """ Row Wise Comparison:"""
    Distance = distance.pdist(Data[Sel_sensors,:,i],'seuclidean')
    """ Column Wise Comparison:"""
    #Distance = distance.pdist(np.transpose(Data[:,:,i]),'seuclidean')

    Similarity_Mat = distance.squareform(Distance)
    Degree = np.sum(Similarity_Mat,0)

    """ Row Wise Comparison:"""
    Degree_Mat = Degree*np.identity(len(Sel_sensors))
    """ Column Wise Comparison:"""
    #Degree_Mat = Degree*np.identity(Data.shape[1])

    Laplacian_Mat = Degree_Mat-Similarity_Mat
    Normalized_Laplacian = (Degree**(-1/2))*Laplacian_Mat*(Degree**(-1/2))
    eig_val = np.linalg.eigvalsh(Normalized_Laplacian)
    FF_Eig = np.array([eig_val[0:5]])
    if len(Eig_Value)==0:
        Eig_Value = FF_Eig
    else:
        Eig_Value = np.concatenate((Eig_Value, FF_Eig))
elapsed = time.time() - t
print('The Calculation time in Second is:', elapsed)
np.save('True_Mutilation_SelSensors_EigVAl_RowWise.npy',Eig_Value)