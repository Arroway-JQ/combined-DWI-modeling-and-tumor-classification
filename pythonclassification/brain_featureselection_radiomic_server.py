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
print("Import library done.\n")


#%% feature selection method2---基于学习模型的特征排序 循环每个特征 需要5次5折交叉验证
datapreparing = joblib.load("./featureselection/datapreparing_radiomics.pkl")
traindataall = datapreparing['traindataall']
WHO = traindataall['WHO']
U_data = traindataall['U_data']
U_featurenames = traindataall['U_featurenames']
print ("Datapreparing done.\n")
# LR
#LogisticRegressionCV 用交叉验证决定C,LogisticRegression好像要自己决定
LRscores = []
model_LRCVs = {}

for i in range(U_data.shape[1]):
     #如果只有一个特征的话要进行.reshape(-1,1)如果只有一个样本的话要进行.reshape(1,-1)
    LRCV = LogisticRegressionCV(cv=5, random_state=0)
    score = cross_val_score(LRCV, U_data[:,i].reshape(-1,1), WHO, scoring = 'accuracy', cv = ShuffleSplit(n_splits = 5,test_size=0.1,random_state=0),n_jobs=-1)
    LRscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_LRCV  = LogisticRegressionCV(cv=5, random_state=0,n_jobs=5).fit(U_data[:,i].reshape(-1,1),WHO)
    model_LRCVs[U_featurenames[i]] = model_LRCV
    print("LR %d done. "%(i))
LRscores_sorted = sorted(LRscores, reverse = True)
LRfeatureselection = {'LRscores_sorted':LRscores_sorted,'model_LRCVs':model_LRCVs}
joblib.dump(LRfeatureselection,'./featureselection/radiomics/LRfeatureselection.pkl')

#%% SVM
SVMscores = []
model_SVMs = {}
param_grid_svm = [{'kernel': ['linear'], 'C': [0.03,0.05,0.1, 0.5]},{'kernel':['rbf'], 'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,10,50,100,1000],'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}]
for i in range(U_data.shape[1]):
    gsearch = GridSearchCV(svm.SVC(),param_grid_svm, scoring='accuracy', cv=10, n_jobs=-1).fit(U_data[:,1].reshape(-1,1),WHO)
    best_parameters = gsearch.best_params_
    ker = best_parameters['kernel']
    bestC = best_parameters['C']
    bestgamma = best_parameters['gamma']
    SVMCV = svm.SVC(C=bestC, kernel=ker, gamma=bestgamma,decision_function_shape='ovo',random_state=0)
    score = cross_val_score(SVMCV, U_data[:,i].reshape(-1,1), WHO, scoring='accuracy', cv=ShuffleSplit(n_splits = 5,test_size=0.1,random_state=0),n_jobs=-1)
    SVMscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_SVM =  svm.SVC(C=bestC, kernel=ker, gamma=bestgamma,decision_function_shape='ovo',random_state=0).fit(U_data[:,i].reshape(-1,1),WHO)
    model_SVMs[U_featurenames[i]] = model_SVM   #后期用model_SVM.get_params()应该就可以得到对应模型的参数了
    print("SVM %d done. "%(i))
SVMscores_sorted = sorted(SVMscores, reverse = True)
SVMfeatureselection = {'SVMscores_sorted':SVMscores_sorted,'model_SVMs':model_SVMs}
joblib.dump(SVMfeatureselection, './featureselection/radiomics/SVMfeatureselection.pkl')

#%% KNN    #‘brute' N小的时候使用，N<30大的话可能就无效了；‘kd_tree',N大D小的时候使用，D大的话可能就完了；’ball_tree‘，N大D大的时候使用, 默认auto可以自动选择
# leaf_size 可以选择默认=30 如果 等于 N的话就实际上是brute
KNNscores = []
model_KNNs = {}
param_grid_knn = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'algorithm':['ball_tree','kd_tree','brute'],'leaf_size':[30,35,40,45,50]}
for i in range(U_data.shape[1]):
    gsearch = GridSearchCV(estimator=KNeighborsClassifier(p=2, metric='minkowski'),param_grid=param_grid_knn, cv=10, scoring='accuracy',n_jobs=-1).fit(U_data[:,i].reshape(-1,1),WHO)
    best_parameters = gsearch.best_params_
    bestn = best_parameters['n_neighbors']
    bestalgo = best_parameters['algorithm']
    bestleaf = best_parameters['leaf_size']
    KNNCV = KNeighborsClassifier(n_neighbors=bestn, algorithm=bestalgo, leaf_size=bestleaf, p=2, metric='minkowski')
    score = cross_val_score(KNNCV, U_data[:,i].reshape(-1,1), WHO, scoring='accuracy', cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=0),n_jobs=-1)
    KNNscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_KNN = KNeighborsClassifier(n_neighbors=bestn, algorithm=bestalgo, leaf_size=bestleaf, p=2, metric='minkowski').fit(U_data[:,i].reshape(-1,1), WHO)
    model_KNNs[U_featurenames[i]] = model_KNN # 后期用model_KNN.get_params() 应该就可以得到对应模型的参数了
    print("KNN %d done. " % (i))
KNNscores_sorted = sorted(KNNscores, reverse=True)
KNNfeatureselection = {'KNNscores_sorted':KNNscores_sorted,'model_KNNs':model_KNNs}
joblib.dump(KNNfeatureselection,'./featureselection/radiomics/KNNfeatureslection.pkl')

#%% RF RF这个调参啊。。。但是好像因为我现在只使用一个特征所以不用做name复杂的调参，么得选，调一下n_estimator（几棵树）和max_depth（树的深度）？这个好像也不怎么需要
RFscores = []
model_RFs = {}
param_grid_RF = {'n_estimators':[5,50,100,1000],'max_depth':[2,4,10]}
for i in range(U_data.shape[1]):
    gsearch = GridSearchCV(estimator=RandomForestClassifier(oob_score=True,random_state=0),param_grid=param_grid_RF, cv=10, scoring='accuracy',n_jobs=-1).fit(U_data[:,i].reshape(-1,1),WHO)
    best_parameters = gsearch.best_params_
    bestrfn = best_parameters['n_estimators']
    bestdepth = best_parameters['max_depth']
    RFCV = RandomForestClassifier(n_estimators=bestrfn, max_depth=bestdepth,oob_score=True, random_state=0)
    score = cross_val_score(RFCV, U_data[:,i].reshape(-1,1), WHO, scoring='accuracy', cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=0),n_jobs=-1)
    RFscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_RF = RandomForestClassifier(n_estimators=bestrfn, max_depth=bestdepth, oob_score=True, random_state=0).fit(U_data[:,i].reshape(-1,1),WHO)
    model_RFs[U_featurenames[i]] = model_RF #后期使用model_RF.get_params()应该就可以得到对应模型的参数了
    print("RF %d done. " % (i))
RFscores_sorted = sorted(RFscores, reverse = True)
RFfeatureselection1 = {'RFscores_sorted':RFscores_sorted, 'model_RFs':model_RFs}
joblib.dump(RFfeatureselection1, './featureselection/radiomics/RFfeatureselection1.pkl')

#  NB 哇这个好简单根本不用调参，哈哈哈
NBscores = []
model_NBs = {}
for i in range(U_data.shape[1]):
    NBCV = GaussianNB()
    score = cross_val_score(NBCV, U_data[:,i].reshape(-1,1), WHO, scoring='accuracy', cv=ShuffleSplit(n_splits=5,test_size=0.1, random_state=0),n_jobs=-1)
    NBscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_NB = GaussianNB().fit(U_data[:,i].reshape(-1,1), WHO)
    model_NBs[U_featurenames[i]] = model_NB
    print("NB %d done. " % (i))
NBscores_sorted = sorted(NBscores, reverse = True)
NBfeatureselection = {'NBscores_sorted':NBscores_sorted, 'model_NBs':model_NBs}
joblib.dump(NBfeatureselection, './featureselection/radiomics/NBfeatureselection.pkl')

# stacking oh 这个需要使用上面几个方法的参数略微比较麻烦一点哈
Stackscores = []
model_stacks = {}
for i in range (U_data.shape[1]):
    clf1 = GaussianNB()
    clf2param = model_RFs[U_featurenames[i]].get_params()
    clf2 = RandomForestClassifier(n_estimators=clf2param['n_estimators'], max_depth=clf2param['max_depth'],oob_score=True, random_state=0)
    clf3param = model_KNNs[U_featurenames[i]].get_params()
    clf3 = KNeighborsClassifier(n_neighbors=clf3param['n_neighbors'], algorithm=clf3param['algorithm'],leaf_size=clf3param['leaf_size'],p=2,metric='minkowski')
    clf4param = model_SVMs[U_featurenames[i]].get_params()
    clf4 = svm.SVC(C=clf4param['C'], kernel=clf4param['kernel'], gamma=clf4param['gamma'], decision_function_shape='ovo', random_state=0)
    estimator = [('NB',clf1),('RF',clf2),('KNN',clf3),('SVM',clf4)]
    clf5 = StackingClassifier(estimators=estimator,final_estimator=LogisticRegressionCV(cv=5,random_state=0),stack_method='auto', n_jobs=-1)
    score = cross_val_score(clf5, U_data[:,i].reshape(-1,1), WHO, scoring='accuracy', cv=ShuffleSplit(n_splits=5, test_size=0.1, random_state=0),n_jobs=-1)
    Stackscores.append((round(np.mean(score),4),U_featurenames[i]))
    model_stack = clf5.fit(U_data[:,i].reshape(-1,1),WHO)
    model_stacks[U_featurenames[i]] = model_stack
    print("Stack %d done. " % (i))
Stackscores_sorted = sorted(Stackscores, reverse=True)
Stackfeatureselection = {'Stackscores':Stackscores_sorted,'model_stacks':model_stacks}
joblib.dump(Stackfeatureselection,'./featureselection/radiomics/Stackfeatureselection.pkl')

#开始统计打分结果--------------1、选特征
KNNfeatures = []
allmodelsfeatures = []
for i in range (len(KNNscores_sorted)):
    KNNfeatures.append(KNNscores_sorted[i][1])
allmodelsfeatures.append(KNNfeatures)

LRfeatures = []
for i in range(len(LRscores_sorted)):
    LRfeatures.append(LRscores_sorted[i][1])
allmodelsfeatures.append(LRfeatures)

NBfeatures = []
for i in range(len(NBscores_sorted)):
    NBfeatures.append(NBscores_sorted[i][1])
allmodelsfeatures.append(NBfeatures)

RFfeatures1 = []
for i in range(len(RFscores_sorted)):
    RFfeatures1.append(RFscores_sorted[i][1])
allmodelsfeatures.append((RFfeatures1))

SVMfeatures = []
for i in range(len(SVMscores_sorted)):
    SVMfeatures.append(SVMscores_sorted[i][1])
allmodelsfeatures.append(SVMfeatures)

Stackfeatures = []
for i in range(len(Stackscores_sorted)):
    Stackfeatures.append(Stackscores_sorted[i][1])
allmodelsfeatures.append(Stackfeatures)
#开始统计打分结果--------------2、选单个特征的模型
sigfeaturemod = {}
for i in range(len(KNNscores)):
    featurename = KNNscores[i][1]
    modelscores = []
    KNNscore = (KNNscores[i][0],'model_KNNs')
    modelscores.append(KNNscore)
    LRscore = (LRscores[i][0],'model_LRCVs')
    modelscores.append((LRscore))
    NBscore = (NBscores[i][0],'model_NBs')
    modelscores.append(NBscore)
    RFscore = (RFscores[i][0],'model_RFs')
    modelscores.append(RFscore)
    SVMscore = (SVMscores[i][0],'model_SVMs')
    modelscores.append(SVMscore)
    Stackscore = (Stackscores[i][0],'model_Stacks')
    modelscores.append(Stackscore)
    modelscores_sorted = sorted(modelscores, reverse=True)
    sigfeaturemod[featurename] = modelscores_sorted
# 保存feature selection method2---基于学习模型的特征排序的结果
joblib.dump(allmodelsfeatures, './featureselection/radiomics/allmodelsfeatures.pkl')
joblib.dump(sigfeaturemod, './featureselection/radiomics/sigfeaturemod.pkl')

#%%---feature selection method3---顶层特征选择-lassoCV
def seq(start, end, step):
    if step ==0:
        raise ValueError("Step must not be 0")
    sample_count = int(abs(end - start)/step)
    return itertools.islice(itertools.count(start, step), sample_count)
for j in seq(0,3,0.0001):
    model_Lasso = LassoCV(alphas = [j],cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0), n_jobs=-1, max_iter=100000).fit(U_data,WHO)
    coef = model_Lasso.coef_
    pos = np.where(abs(coef)>0.0001)
    pos = np.array(pos)
    pos= pos[0,:]
    if len(pos) == 20:
        break
#%%整理Lasso选特征的结果
Lassoscores = []
Lassoscore = abs(coef[pos])
Lassofeature = U_featurenames[pos]
for i in range(len(pos)):
    Lassoscores.append((round(Lassoscore[i],4), pos[i], Lassofeature[i]))
Lassoscores_sorted = sorted(Lassoscores, key=lambda x: x[0], reverse=True)
Lassofeatures = []
for i in range(len(Lassoscores_sorted)):
    Lassofeatures.append(Lassoscores_sorted[i][2])
for i in range(20,2289,1):
    Lassofeatures.append('None')
allmodelsfeatures.append(Lassofeatures)
Lassofeatureselection = {'Lassoscores':Lassoscores_sorted,'model_Lasso':model_Lasso}

joblib.dump(Lassofeatureselection, './featureselection/radiomics/Lassofeatureselection.pkl')
joblib.dump(allmodelsfeatures, './featureselection/radiomics/allmodelsfeatures.pkl')
print("Lasso scroing done. \n")
#%%---feature selection method4---RandomForest整体特征调参选特征
param_grid_RF2 = {'n_estimators':[100,1000,10000],'max_depth':[5,10,20,50,100], 'min_samples_split':[2,4,6,8,10,20,30], 'min_samples_leaf':[1,2,3,4,5,10,15,20,25,30], 'max_features':[3,5,10,15,20,25,30]}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(oob_score=True,random_state=0),param_grid=param_grid_RF2, cv=10, scoring='accuracy',n_jobs=-1).fit(U_data,WHO)
bestn2 = gsearch2.best_params_['n_estimators']
bestdepth2 = gsearch2.best_params_['max_depth']
bestminsplit = gsearch2.best_params_['min_samples_split']
bestminleaf = gsearch2.best_params_['min_samples_leaf']
bestmaxfeat = gsearch2.best_params_['max_features']
model_RF2 = RandomForestClassifier(n_estimators=bestn2, max_depth=bestdepth2, min_samples_split=bestminsplit, min_samples_leaf=bestminleaf,max_features=bestmaxfeat, oob_score=True, random_state=0,n_jobs=-1).fit(U_data,WHO)
importances = model_RF2.feature_importances_
indices = np.argsort(importances)[::-1]

RFscores2_sorted = []
RFfeatures2 = []
for i in range(U_data.shape[1]):
    RFscores2_sorted.append((importances[indices[i]],U_featurenames[indices[i]], indices[i]))
for i in range(len(RFscores2_sorted)):
    RFfeatures2.append(RFscores2_sorted[i][1])
allmodelsfeatures.append(RFfeatures2)
RFfeatureselection2 = {'RFscores2':RFscores2_sorted, 'model_RF2':model_RF2}

joblib.dump(RFfeatureselection2,'./featureselection/radiomics/RFfeatureselection2.pkl')
joblib.dump(allmodelsfeatures, './featureselection/radiomics/allmodelsfeatures.pkl')
print("RF2 scoring done. \n")
#%%---feature selection method5---递归特征消除，RFE，使用Ridge 稳定
#先对Ridge进行调参
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import RFECV # scoring不能使用 ‘accuracy’ 因为回归输出的是0，1
model_Ridge = RidgeCV(alphas=[0.01, 0.1, 10, 100], scoring='r2', cv=ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)).fit(U_data,WHO)
bestalpha = model_Ridge.alpha_
model_ridge = Ridge(alpha = bestalpha, random_state=0, max_iter=10000, tol=1e-4)
model_RFE = RFECV(estimator=model_ridge, step=1, min_features_to_select=1, cv=ShuffleSplit(n_splits=5, test_size=0.1, random_state=0), scoring='r2', n_jobs=-1).fit(U_data,WHO)
RFEfeaturenum = model_RFE.n_features_
RFEscores = model_RFE.ranking_
RFEscores_sorted = sorted(zip(map(lambda x: round(x,4), RFEscores), U_featurenames))
RFEfeatureselection = {'RFEscores':RFEscores_sorted, 'model_RFE': model_RFE, 'RFEfeaturenum': RFEfeaturenum}
RFEfeatures = []
for i in range(len(RFEscores_sorted)):
	RFEfeatures.append(RFEscores_sorted[i][1])
allmodelsfeatures = []
allmodelsfeatures.append(KNNfeatures)
allmodelsfeatures.append(LRfeatures)
allmodelsfeatures.append(NBfeatures)
allmodelsfeatures.append(RFfeatures1)
allmodelsfeatures.append(SVMfeatures)
allmodelsfeatures.append(Stackfeatures)
allmodelsfeatures.append(Lassofeatures)
allmodelsfeatures.append(RFfeatures2)
allmodelsfeatures.append(RFEfeatures)

joblib.dump(RFEfeatureselection, './featureselection/radiomics/RFEfeatureselection.pkl')
joblib.dump(allmodelsfeatures, './featureselection/radiomics/allmodelsfeatures.pkl')
print("allmodelsfeatures done.\n")

#%% voting system
import pandas as pd
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
