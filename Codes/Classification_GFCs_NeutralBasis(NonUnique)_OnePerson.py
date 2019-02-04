import numpy as np
import os
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
# Setting the path
os.chdir('D:\\Algorithms\\NeuroScience\\Updated SpaceCat\\Data\\True')
#Loading the data
""" ColumnWise Comp.: """
#Neutral_EIG = np.load('UpdatedData_True_Neutral_GFC_ColumnWise.npy')
#Mutilation_EIG = np.load('UpdatedData_True_Mutilation_GFC_ColumnWise.npy')
""" RowWise Comp.:"""
#Neutral_EIG = np.load('UpdatedData_True_Neutral_GFC_RowWise.npy')
#Mutilation_EIG = np.load('UpdatedData_True_Mutilation_GFC_RowWise.npy')

Neutral_EIG = np.load('UpdatedData_LeastBluery_True_Neutral_GFC(NonUnique_NaturalBasis)_RowWise.npy')
Mutilation_EIG = np.load('UpdatedData_LeastBluery_True_Mutilation_GFC(NonUnique_NaturalBasis)_RowWise.npy')
Trial_No_Neutral = np.load('TrialNo_True_Neutral_LeastBlury.npy')
Trial_No_Mutilation = np.load('TrialNo_True_Mutilation_LeastBlury.npy')
#Data Preparation
Neutral_EIG = np.transpose(Neutral_EIG)         # This is to put all GFCs for an image in a row
Mutilation_EIG = np.transpose(Mutilation_EIG)   # This is to put all GFCs for an image in a row
n_splits = 5
kf = KFold(n_splits,shuffle=True)
Replication_no = 50
AllRep_FScore = np.zeros((1,Replication_no))
AllSubj_FScore = np.zeros((1,len(Trial_No_Neutral)))
for i in range(len(Trial_No_Neutral)):
    Sel_Neutral_EIG = Neutral_EIG[int(np.sum(Trial_No_Neutral[:i]))+1 : int(np.sum(Trial_No_Neutral[:i+1]))+1,:]
    Sel_Mutilation_EIG = Mutilation_EIG[int(np.sum(Trial_No_Mutilation[:i]))+1 : int(np.sum(Trial_No_Mutilation[:i+1]))+1,:]
    # Seperation of data into train and test set
    N_len = len(Sel_Neutral_EIG)
    M_len = len(Sel_Mutilation_EIG)
    Neutral_lbl, Mutilation_lbl = np.zeros((N_len,)), np.ones(((M_len),)) 
    X = np.concatenate((Sel_Neutral_EIG, Sel_Mutilation_EIG))
    Y = np.concatenate((Neutral_lbl, Mutilation_lbl))
    F_score = np.zeros((1,n_splits))
    for k in range(Replication_no):
        counter = 0    #counter for f_score and confusionnmat of each fold
        conf_mat = np.array([[0,0],[0,0]])
        for train_ind, test_ind in kf.split(X):
            X_Train, X_Test = X[train_ind], X[test_ind]
            Y_Train, Y_Test = Y[train_ind], Y[test_ind]
            clf = svm.LinearSVC()
            clf.fit(X_Train,Y_Train)
            Y_Predict = clf.predict(X_Test)
            conf_mat = conf_mat + confusion_matrix(Y_Test,Y_Predict)
            F_score[0,counter] = f1_score(Y_Test,Y_Predict)
            counter += 1
        Pr = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1])
        Re = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0])
        AllRep_FScore[0,k] = 2*(Pr*Re)/(Pr+Re)
        AllRep_FScore[0,k] = np.mean(F_score)
        print(conf_mat)
    AllSubj_FScore [0,i] =np.mean(AllRep_FScore)   
    print('F_Score of Subject', i, '=', np.mean(AllRep_FScore))