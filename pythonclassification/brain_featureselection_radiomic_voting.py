#%% load liabrary
import os
import numpy as np
import xlrd
import scipy.io as io
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.model_selection import cross_val_score,ShuffleSplit, train_test_split
from sklearn.linear_model import Lasso, Ridge, LassoCV,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn import metrics
import itertools
import joblib
from scipy import stats
import pandas as pd
print("Import library done.\n")

#%% feature selection method2---基于学习模型的特征排序 循环每个特征 需要5次5折交叉验证
datapreparing = joblib.load("./featureselection/datapreparing_radiomics.pkl")
traindataall = datapreparing['traindataall']
WHO = traindataall['WHO']
U_data = traindataall['U_data']
U_featurenames = traindataall['U_featurenames']
allmodelsfeatures = joblib.load('./featureselection/radiomics/allmodelsfeatures.pkl')
Lassofeatureselection = joblib.load('./featureselection/radiomics/Lassofeatureselection.pkl')
Lassoscores_sorted = Lassofeatureselection['Lassoscores']
Lassofeatures = []
for i in range(len(Lassoscores_sorted)):
    Lassofeatures.append(Lassoscores_sorted[i][2])
for i in range(20,2289,1):
    Lassofeatures.append('None')
allmodelsfeatures.append(Lassofeatures)

RFfeatureselection2 = joblib.load('./featureselection/radiomics/RFfeatureselection2.pkl')
RFscores2_sorted = RFfeatureselection2['RFscores2']
RFfeatures2 = []
for i in range(len(RFscores2_sorted)):
    RFfeatures2.append(RFscores2_sorted[i][1])
allmodelsfeatures.append(RFfeatures2)

RFEfeatureselection = joblib.load('./featureselection/radiomics/RFEfeatureselection.pkl')
RFEscores_sorted = RFEfeatureselection['RFEscores']
RFEfeatures = []
for i in range(len(RFEscores_sorted)):
	RFEfeatures.append(RFEscores_sorted[i][1])
allmodelsfeatures.append(RFEfeatures)
joblib.dump(allmodelsfeatures, './featureselection/radiomics/allmodelsfeatures.pkl')
print ("Datapreparing done.\n")

#%% voting system
votingfeatures = allmodelsfeatures[8][0:20]
votes = [1]*20
for i in range(len(allmodelsfeatures)-1):
    sameindex = []
    operationfeatures = allmodelsfeatures[i][0:20]
    same = [x for x in votingfeatures if x in operationfeatures]
    for name in same:
        sameindex.append(votingfeatures.index(name))
    for p in sameindex:
        votes[p] = votes[p]+1
    none = [y for y in operationfeatures if y not in votingfeatures]
    votingfeatures.extend(none)
    nonenum = len(none)
    newvotes = [1]*nonenum
    votes.extend(newvotes)
asist = [0]*len(votes)
votes=[votes,asist]
votingresults = pd.DataFrame(votes,columns=votingfeatures)
joblib.dump(votingresults,'./featureselection/radiomics/votingresults_radiomics.pkl')
votingresults.to_excel('./featureselection/radiomics/votingresults.xlsx')
print("voting system done.\n")
#%%---将allmodelsfeatures 导出到Excel里面总结出选择的特征
df1 = pd.DataFrame(allmodelsfeatures)
df1.to_excel('./featureselection/radiomics/allmodelsfeatures.xlsx')
