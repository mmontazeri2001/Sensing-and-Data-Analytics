import numpy as np
import os
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# Setting the path
os.chdir('D:\\Algorithms\\NeuroScience\\Updated SpaceCat\\Data\\True')
#Loading the data
""" ColumnWise Comp.: """
#Neutral_EIG = np.load('UpdatedData_True_Neutral_GFC_ColumnWise.npy')
#Mutilation_EIG = np.load('UpdatedData_True_Mutilation_GFC_ColumnWise.npy')
""" RowWise Comp.:"""
Neutral_EIG = np.load('UpdatedData_True_Neutral_GFC_RowWise.npy')
Mutilation_EIG = np.load('UpdatedData_True_Mutilation_GFC_RowWise.npy')
#Neutral_EIG = np.load('UpdatedData_LeastBluery_True_Neutral_GFC_RowWise.npy')
#Mutilation_EIG = np.load('UpdatedData_LeastBluery_True_Mutilation_GFC_RowWise.npy')

Neutral_EIG = np.transpose(Neutral_EIG)         # This is to put all GFCs for an image in a row
Mutilation_EIG = np.transpose(Mutilation_EIG)   # This is to put all GFCs for an image in a row

#Data Preparation
Train_Rate = 0.8
N_len = len(Neutral_EIG)
M_len = len(Mutilation_EIG)
Neutral_lbl = np.zeros((N_len,))
Mutilation_lbl = np.ones(((M_len),))
X_Train = np.concatenate((Neutral_EIG[:round(N_len*Train_Rate)], Mutilation_EIG[:round(M_len*Train_Rate)]))
X_Test  = np.concatenate((Neutral_EIG[round(N_len*Train_Rate):], Mutilation_EIG[round(M_len*Train_Rate):]))
Y_Train = np.concatenate((Neutral_lbl[:round(N_len*Train_Rate)], Mutilation_lbl[:round(M_len*Train_Rate)]))
Y_Test  = np.concatenate((Neutral_lbl[round(N_len*Train_Rate):], Mutilation_lbl[round(M_len*Train_Rate):]))
clf = svm.LinearSVC()
clf.fit(X_Train,Y_Train)
Y_Predict = clf.predict(X_Test)
print(confusion_matrix(Y_Test,Y_Predict))
print(f1_score(Y_Test,Y_Predict))