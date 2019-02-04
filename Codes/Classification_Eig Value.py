import numpy as np
import os
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
# Setting the path
os.chdir('D:\\Algorithms\\NeuroScience\\Updated SpaceCat\\Data\\True')
#Loading the data
Neutral_EIG = np.load('True_Neutral_SelSensors_LeastBlury_EigVAl_RowWise.npy')
Mutilation_EIG = np.load('True_Mutilation_SelSensors_LeastBlury_EigVAl_RowWise.npy')
#Data Preparation
"""Train_Rate = 0.8
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
print(f1_score(Y_Test,Y_Predict))"""
Trial_No_Neutral = np.load('TrialNo_True_Neutral_LeastBlury.npy')
Trial_No_Mutilation = np.load('TrialNo_True_Mutilation_LeastBlury.npy')
#Data Preparation
n_splits = 5
kf = KFold(n_splits,shuffle=True)
Sel_Neutral_EIG = Neutral_EIG       # I just used the code in OnePerson version so directly put all the EIG data in this variable.
Sel_Mutilation_EIG = Mutilation_EIG
# Seperation of data into train and test set
N_len = len(Sel_Neutral_EIG)
M_len = len(Sel_Mutilation_EIG)
Neutral_lbl, Mutilation_lbl = np.zeros((N_len,)), np.ones(((M_len),)) 
X = np.concatenate((Sel_Neutral_EIG, Sel_Mutilation_EIG))
Y = np.concatenate((Neutral_lbl, Mutilation_lbl))
conf_mat = {}
F_score = np.zeros((1,n_splits))
All_Scores = np.zeros((1,100))
for j in range(100):
    counter = 0    #counter for f_score and confusionnmat of each fold
    for train_ind, test_ind in kf.split(X):
        X_Train, X_Test = X[train_ind], X[test_ind]
        Y_Train, Y_Test = Y[train_ind], Y[test_ind]
        clf = svm.LinearSVC()
        clf.fit(X_Train,Y_Train)
        Y_Predict = clf.predict(X_Test)
        #conf_mat[counter] = confusion_matrix(Y_Test,Y_Predict)
        #print(confusion_matrix(Y_Test,Y_Predict))
        F_score[0,counter] = f1_score(Y_Test,Y_Predict)
        counter += 1  
    All_Scores [0,j] = np.mean(F_score)
    #print('F_Score of replication', j, '=', np.mean(F_score))
print('Mean of All F_Scores =', np.mean(All_Scores))
print('Std Dev of All F_Scores =', np.std(All_Scores))