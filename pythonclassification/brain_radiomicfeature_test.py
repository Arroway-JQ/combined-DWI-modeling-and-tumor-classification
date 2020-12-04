#%% load liabrary
import os
import numpy as np

import xlrd
import scipy.io as io
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.linear_model import Lasso, Ridge, LassoCV,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import itertools
import matplotlib.pyplot as plt
import joblib
import subprocess
print("Import library done.\n")

#%% ------------featuregroup 的最终在externel-Renji testset上的结果（用平均AUC选模）
ALLgroups = joblib.load("./estimatorprediction/ALLgroups_radiomics.pkl")
ALLgroup_bestmodscore = ALLgroups['ALLgroup_bestmodscore']
ALLgroup_bestmodmeanscore_sorted = ALLgroups['ALLgroup_bestmodmeanscore']
ALLgroupresults = ALLgroups['ALLgroupresults']
select_features = joblib.load("./estimatorprediction/select_features_radiomics.pkl")
datapreparing = joblib.load('./featureselection/datapreparing_radiomics.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
U_featurenames = traindataall['U_featurenames'].tolist()
U_data = traindataall['U_data']
U_data_test = testdataall['U_data_test']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
featurenames = traindataall['featurenames']
data_test = testdataall['data']
print("AUC Load prepared data done.")
#%%
allgroups_testaccs2 = []
allgroups_testresults2 = {}
groups2_testaccs2 = [] #第一个数字2 表示在这个组合中一共使用了2个feature， 第二个数字表示选模的第二种（计算平均AUC）的方法
groups2_testresults2 = {}
groups3_testaccs2 = []
groups3_testresults2 = {}
groups4_testaccs2 = []
groups4_testresults2 = {}
groups5_testaccs2 = []
groups5_testresults2 = {}
groups6_testaccs2 = []
groups6_testresults2 = {}
groups7_testaccs2 = []
groups7_testresults2 = {}
groups8_testaccs2 = []
groups8_testresults2 = {}
groups9_testaccs2 = []
groups9_testresults2 = {}

for key in ALLgroupresults.keys():
    allmodmeanaucs = []
    featurename = key
    simplefeaturename = featurename.replace('select_features_','')
    val = ALLgroupresults[key]
    ALLmod = val['ALLmod']
    ALLmodmeanscores_sorted = val['ALLmodmeanscores']
    LRmodels = {}
    SVMmodels = {}
    KNNmodels = {}
    NBmodels = {}
    RFmodels = {}
    Stackmodels = {}
    select_features_mod = select_features[key]
    select_U_features_mod = list(select_features_mod.values())[0]
    select_U_data_test_mod = list(select_features_mod.values())[2]
    select_U_data_mod = list(select_features_mod.values())[1]
    for K in ALLmod.keys():
        if 'LR' in K:
            LRmodels[K] = ALLmod[K]
        elif 'SVM' in K:
            SVMmodels[K] = ALLmod[K]
        elif 'KNN' in K:
            KNNmodels[K] = ALLmod[K]
        elif 'NB' in K:
            NBmodels[K] = ALLmod[K]
        elif 'RF' in K:
            RFmodels[K] = ALLmod[K]
        elif 'Stack' in K:
            Stackmodels[K] = ALLmod[K]
    LRtprs = []  #用来绘制平均ROC
    LRaucs = []
    SVMtprs = []
    SVMaucs = []
    KNNtprs = []
    KNNaucs = []
    NBtprs = []
    NBaucs = []
    RFtprs = []
    RFaucs = []
    Stacktprs = []
    Stackaucs = []

    LRrocs = [] #用来保存每一个fold下的tpr和fpr绘制每一个fold下的ROC曲线
    SVMrocs = []
    KNNrocs = []
    NBrocs = []
    RFrocs = []
    Stackrocs = []

    LRmean_fpr = np.linspace(0,1,100)
    SVMmean_fpr = np.linspace(0,1,100)
    KNNmean_fpr = np.linspace(0,1,100)
    NBmean_fpr = np.linspace(0,1,100)
    RFmean_fpr = np.linspace(0,1,100)
    Stackmean_fpr = np.linspace(0,1,100)

    LRscores = []
    SVMscores = []
    KNNscores = []
    NBscores = []
    RFscores = []
    Stackscores = []

    LRscores_th = []
    SVMscores_th = []
    KNNscores_th = []
    NBscores_th = []
    RFscores_th = []
    Stackscores_th = []


    for k in LRmodels.keys():
        modname = k
        modelresults = LRmodels[k]
        LRCV = modelresults['LRCV']
        X_train = modelresults['X_train']
        Y_train = modelresults['Y_train']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        LRscore = LRCV.score(X_test,Y_test)
        Y_pred = LRCV.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr,tpr)
        if roc_auc <0.5:
            Y_pred = LRCV.predict_proba(X_test)[:,0]
            fpr, tpr, thresholds = roc_curve(Y_test,Y_pred)
            roc_auc = auc(fpr,tpr)
        Y_pred2 = []
        optimal_idxf = np.argmax(tpr-fpr)
        optimal_thresholdsf = thresholds[optimal_idxf]
        for prob in Y_pred:
            if prob >= optimal_thresholdsf:
                Y_pred2.append(1)
            else:
                Y_pred2.append(0)
        LRscore_th = accuracy_score(Y_test,Y_pred2)
        LRtprs.append(np.interp(LRmean_fpr, fpr, tpr))
        LRtprs[-1][0] = 0.0
        LRaucs.append(roc_auc)
        LRrocs.append((fpr,tpr,roc_auc,modname,optimal_thresholdsf))
        LRscores.append(LRscore)
        LRscores_th.append(LRscore_th)
    LRmeanscore = (round(np.mean(LRscores),4), 'LRmeanscore')
    LRmeanscore_th = (round(np.mean(LRscores_th),4),'LRmeanscore_th')
    LRrocs_sorted = sorted(LRrocs,key=lambda x:x[2],reverse=True)
    LRmodname = LRrocs_sorted[0][3]
    optimal_thresholdsf = LRrocs_sorted[0][4]
    LRmodel = LRmodels[LRmodname]['LRCV']
    WHO_trainpred = LRmodel.predict_proba(select_U_data_mod)[:,1]
    fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    roc_auch = auc(fprh,tprh)
    if roc_auch < 0.5:
        WHO_trainpred = LRmodel.predict_proba(select_U_data_mod)[:,0]
        fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    optimal_idx = np.argmax(tprh-fprh)
    optimal_thresholdsh = thresholdsh[optimal_idx]
    LRmean_tpr = np.mean(LRtprs, axis=0)
    LRmean_tpr[-1] = 1.0
    LRmean_auc = auc(LRmean_fpr, LRmean_tpr)
    LRstd_auc = np.std(LRaucs,ddof=1)
    LRstd_tpr = np.std(LRtprs, axis=0)
    LRtprs_upper = np.minimum(LRmean_tpr + LRstd_tpr, 1)
    LRtprs_lower = np.maximum(LRmean_tpr - LRstd_tpr, 0)
    LRrocresults = {'LRmodels':LRmodels, 'LRrocs':LRrocs_sorted, 'LRmean_tpr':LRmean_tpr, 'LRmean_fpr':LRmean_fpr, 'LRtprs_upper':LRtprs_upper, 'LRtprs_lower':LRtprs_lower,'optimal_thresholdsh': optimal_thresholdsh,'optimal_thresholdsf':optimal_thresholdsf,'LRmeanscore':LRmeanscore,'LRmeanscore_th':LRmeanscore_th}
    allmodmeanaucs.append((LRmean_auc,'LR'))
    print("\nLR models done.")
    for k in SVMmodels.keys():
        modname = k
        modelresults = SVMmodels[k]
        SVMCV = modelresults['SVMCV']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        X_train = modelresults['X_train']
        Y_train = modelresults['Y_train']
        try:
            Y_pred = SVMCV.predict(X_test)
        except Exception as e:
            params = SVMCV.get_params()
            SVMCV = svm.SVC(C=params['C'],kernel=params['kernel'],gamma=params['gamma'],decision_function_shape='ovo',random_state=0).fit(X_train, Y_train)
            Y_pred = SVMCV.predict(X_test)
        SVMscore = SVMCV.score(X_test,Y_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr,tpr)
        SVMscore_th = SVMscore
        SVMscores.append(SVMscore)
        SVMscores_th.append(SVMscores_th)
        SVMtprs.append(np.interp(SVMmean_fpr, fpr, tpr))
        SVMtprs[-1][0] = 0.0
        SVMaucs.append(roc_auc)
        SVMrocs.append((fpr,tpr,roc_auc,modname))
    SVMrocs_sorted = sorted(SVMrocs,key=lambda x:x[2],reverse=True)  #按照roc_auc来进行排序
    SVMmeanscore = (round(np.mean(SVMscores), 4), 'SVMmeanscore')
    SVMmeanscore_th = SVMmeanscore
    SVMmean_tpr = np.mean(SVMtprs, axis=0)
    SVMmean_tpr[-1] = 1.0
    SVMmean_auc = auc(SVMmean_fpr, SVMmean_tpr)
    SVMstd_auc = np.std(SVMaucs,ddof=1)
    SVMstd_tpr = np.std(SVMtprs, axis=0)
    SVMtprs_upper = np.minimum(SVMmean_tpr + SVMstd_tpr, 1)
    SVMtprs_lower = np.maximum(SVMmean_tpr - SVMstd_tpr, 0)
    SVMrocresults = {'SVMmodels':SVMmodels,'SVMrocs':SVMrocs_sorted, 'SVMmean_tpr':SVMmean_tpr, 'SVMmean_fpr':SVMmean_fpr, 'SVMtprs_upper':SVMtprs_upper, 'SVMtprs_lower':SVMtprs_lower,'SVMmeanscore':SVMmeanscore,'SVMmeanscore_th':SVMmeanscore_th}
    allmodmeanaucs.append((SVMmean_auc,'SVM'))
    print("SVMmodels done.")
    for k in KNNmodels.keys():
        modname = k
        modelresults = KNNmodels[k]
        KNNCV = modelresults['KNNCV']
        X_train = modelresults['X_train']
        Y_train = modelresults['Y_train']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        Y_pred = KNNCV.predict_proba(X_test)[:,1]
        KNNscore = KNNCV.score(X_test,Y_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr,tpr)
        if roc_auc <0.5:
            Y_pred = KNNCV.predict_proba(X_test)[:,0]
            fpr, tpr, thresholds = roc_curve(Y_test,Y_pred)
            roc_auc = auc(fpr,tpr)
        Y_pred2 = []
        optimal_idxf = np.argmax(tpr-fpr)
        optimal_thresholdsf = thresholds[optimal_idxf]
        for prob in Y_pred:
            if prob >= optimal_thresholdsf:
                Y_pred2.append(1)
            else:
                Y_pred2.append(0)
        KNNscore_th = accuracy_score(Y_test,Y_pred2)
        KNNscores.append(KNNscore)
        KNNscores_th.append(KNNscore_th)
        KNNtprs.append(np.interp(KNNmean_fpr, fpr, tpr))
        KNNtprs[-1][0] = 0.0
        KNNaucs.append(roc_auc)
        KNNrocs.append((fpr,tpr,roc_auc,modname,optimal_thresholdsf))
    KNNmeanscore = (round(np.mean(KNNscores), 4), 'KNNmeanscore')
    KNNmeanscore_th = (round(np.mean(KNNscores_th), 4), 'KNNmeanscore_th')
    KNNrocs_sorted = sorted(KNNrocs,key=lambda x:x[2],reverse=True)
    KNNmodname = KNNrocs_sorted[0][3]
    optimal_thresholdsf = KNNrocs_sorted[0][4]
    KNNmodel = KNNmodels[KNNmodname]['KNNCV']
    WHO_trainpred = KNNmodel.predict_proba(select_U_data_mod)[:,1]
    fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    roc_auch = auc(fprh,tprh)
    if roc_auch < 0.5:
        WHO_trainpred = KNNmodel.predict_proba(select_U_data_mod)[:,0]
        fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    optimal_idx = np.argmax(tprh-fprh)
    optimal_thresholdsh = thresholdsh[optimal_idx]
    KNNmean_tpr = np.mean(KNNtprs, axis=0)
    KNNmean_tpr[-1] = 1.0
    KNNmean_auc = auc(KNNmean_fpr, KNNmean_tpr)
    KNNstd_auc = np.std(KNNaucs,ddof=1)
    KNNstd_tpr = np.std(KNNtprs, axis=0)
    KNNtprs_upper = np.minimum(KNNmean_tpr + KNNstd_tpr, 1)
    KNNtprs_lower = np.maximum(KNNmean_tpr -KNNstd_tpr, 0)
    KNNrocresults = {'KNNmodels':KNNmodels, 'KNNrocs':KNNrocs_sorted, 'KNNmean_tpr':KNNmean_tpr, 'KNNmean_fpr':KNNmean_fpr, 'KNNtprs_upper':KNNtprs_upper, 'KNNtprs_lower':KNNtprs_lower,'optimal_thresholdsh':optimal_thresholdsh,'optimal_thresholdsf':optimal_thresholdsf,'KNNmeanscore':KNNmeanscore,'KNNmeanscore_th':KNNmeanscore_th}
    allmodmeanaucs.append((KNNmean_auc,'KNN'))
    print("KNNmodels done.")
    for k in NBmodels.keys():
        modname = k
        modelresults = NBmodels[k]
        NBCV = modelresults['NBCV']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        Y_pred = NBCV.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        NBscore = NBCV.score(X_test,Y_test)
        roc_auc = auc(fpr,tpr)
        if roc_auc <0.5:
            Y_pred = NBCV.predict_proba(X_test)[:,0]
            fpr, tpr, thresholds = roc_curve(Y_test,Y_pred)
            roc_auc = auc(fpr,tpr)
        Y_pred2 = []
        optimal_idxf = np.argmax(tpr-fpr)
        optimal_thresholdsf = thresholds[optimal_idxf]
        for prob in Y_pred:
            if prob >= optimal_thresholdsf:
                Y_pred2.append(1)
            else:
                Y_pred2.append(0)
        NBscore_th = accuracy_score(Y_test, Y_pred2)
        NBscores.append(NBscore)
        NBscores_th.append(NBscore_th)
        NBtprs.append(np.interp(NBmean_fpr, fpr, tpr))
        NBtprs[-1][0] = 0.0
        NBaucs.append(roc_auc)
        NBrocs.append((fpr,tpr,roc_auc,modname,optimal_thresholdsf))
    NBmeanscore = (round(np.mean(NBscores),4),'NBmeanscore')
    NBmeanscore_th = (round(np.mean(NBscores_th), 4), 'NBmeanscore_th')
    NBrocs_sorted = sorted(NBrocs,key=lambda x:x[2],reverse=True)
    NBmodname = NBrocs_sorted[0][3]
    optimal_thresholdsf = NBrocs_sorted[0][4]
    NBmodel = NBmodels[NBmodname]['NBCV']
    WHO_trainpred = NBmodel.predict_proba(select_U_data_mod)[:,1]
    fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    roc_auch = auc(fprh,tprh)
    if roc_auch < 0.5:
        WHO_trainpred = NBmodel.predict_proba(select_U_data_mod)[:,0]
        fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    optimal_idx = np.argmax(tprh-fprh)
    optimal_thresholdsh = thresholdsh[optimal_idx]
    NBmean_tpr = np.mean(NBtprs, axis=0)
    NBmean_tpr[-1] = 1.0
    NBmean_auc = auc(NBmean_fpr, NBmean_tpr)
    NBstd_auc = np.std(NBaucs,ddof=1)
    NBstd_tpr = np.std(NBtprs, axis=0)
    NBtprs_upper = np.minimum(NBmean_tpr + NBstd_tpr, 1)
    NBtprs_lower = np.maximum(NBmean_tpr -NBstd_tpr, 0)
    NBrocresults = {'NBmodels':NBmodels, 'NBrocs':NBrocs_sorted, 'NBmean_tpr':NBmean_tpr, 'NBmean_fpr':NBmean_fpr, 'NBtprs_upper':NBtprs_upper, 'NBtprs_lower':NBtprs_lower,'optimal_thresholdsh':optimal_thresholdsh,'optimal_thresholdsf':optimal_thresholdsf,'NBmeanscore':NBmeanscore,'NBmeanscore_th':NBmeanscore_th}
    allmodmeanaucs.append((NBmean_auc,'NB'))
    print("NB models done.")
    for k in RFmodels.keys():
        modname = k
        modelresults = RFmodels[k]
        RF = modelresults['RF']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        RFscore = RF.score(X_test,Y_test)
        Y_pred = RF.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr,tpr)
        if roc_auc <0.5:
            Y_pred = RF.predict_proba(X_test)[:,0]
            fpr, tpr, thresholds = roc_curve(Y_test,Y_pred)
            roc_auc = auc(fpr,tpr)
        Y_pred2 = []
        optimal_idxf = np.argmax(tpr-fpr)
        optimal_thresholdsf = thresholds[optimal_idxf]
        for prob in Y_pred:
            if prob >= optimal_thresholdsf:
                Y_pred2.append(1)
            else:
                Y_pred2.append(0)
        RFscore_th = accuracy_score(Y_test, Y_pred2)
        RFscores.append(RFscore)
        RFscores_th.append(RFscore_th)
        RFtprs.append(np.interp(RFmean_fpr, fpr, tpr))
        RFtprs[-1][0] = 0.0
        RFaucs.append(roc_auc)
        RFrocs.append((fpr,tpr,roc_auc,modname,optimal_thresholdsf))
    RFmeanscore = (round(np.mean(RFscores),4),'RFmeanscore')
    RFmeanscore_th = (round(np.mean(RFscores_th),4),'RFmeanscore_th')
    RFrocs_sorted = sorted(RFrocs,key=lambda x:x[2],reverse=True)
    RFmodname = RFrocs_sorted[0][3]
    optimal_thresholdsf = RFrocs_sorted[0][4]
    RFmodel = RFmodels[RFmodname]['RF']
    WHO_trainpred = RFmodel.predict_proba(select_U_data_mod)[:,1]
    fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    roc_auch = auc(fprh,tprh)
    if roc_auch < 0.5:
        WHO_trainpred = RFmodel.predict_proba(select_U_data_mod)[:,0]
        fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    optimal_idx = np.argmax(tprh-fprh)
    optimal_thresholdsh = thresholdsh[optimal_idx]
    RFmean_tpr = np.mean(RFtprs, axis=0)
    RFmean_tpr[-1] = 1.0
    RFmean_auc = auc(RFmean_fpr, RFmean_tpr)
    RFstd_auc = np.std(RFaucs,ddof=1)
    RFstd_tpr = np.std(RFtprs, axis=0)
    RFtprs_upper = np.minimum(RFmean_tpr + RFstd_tpr, 1)
    RFtprs_lower = np.maximum(RFmean_tpr - RFstd_tpr, 0)
    RFrocresults = {'RFmodels':RFmodels,'RFrocs':RFrocs_sorted, 'RFmean_tpr':RFmean_tpr, 'RFmean_fpr':RFmean_fpr, 'RFtprs_upper':RFtprs_upper, 'RFtprs_lower':RFtprs_lower,'optimal_thresholdsh':optimal_thresholdsh,'optimal_thresholdsf':optimal_thresholdsf,'RFmeanscore':RFmeanscore,'RFmeanscore_th':RFmeanscore_th}
    allmodmeanaucs.append((RFmean_auc,'RF'))
    print("RFmodels done.")
    for k in Stackmodels.keys():
        modname = k
        modelresults = Stackmodels[k]
        Stack = modelresults['Stack']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        X_train = modelresults['X_train']
        Y_train = modelresults['Y_train']
        try:
            Y_pred = Stack.predict_proba(X_test)[:, 1]
        except Exception as e:
            params = Stack.get_params()
            Stack = StackingClassifier(estimators=params['estimators'], final_estimator=params['final_estimator'], cv=params['cv'], stack_method=params['stack_method'], n_jobs=-1).fit(X_train, Y_train)
            Y_pred = Stack.predict_proba(X_test)[:,1]
        Stackscore = Stack.score(X_test,Y_test)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr,tpr)
        if roc_auc <0.5:
            Y_pred = Stack.predict_proba(X_test)[:,0]
            fpr, tpr, thresholds = roc_curve(Y_test,Y_pred)
            roc_auc = auc(fpr,tpr)
        Y_pred2 = []
        optimal_idxf = np.argmax(tpr-fpr)
        optimal_thresholdsf = thresholds[optimal_idxf]
        for prob in Y_pred:
            if prob >= optimal_thresholdsf:
                Y_pred2.append(1)
            else:
                Y_pred2.append(0)
        Stackscore_th = accuracy_score(Y_test, Y_pred2)
        Stackscores.append(Stackscore)
        Stackscores_th.append(Stackscore_th)
        Stacktprs.append(np.interp(Stackmean_fpr, fpr, tpr))
        Stacktprs[-1][0] = 0.0
        Stackaucs.append(roc_auc)
        Stackrocs.append((fpr,tpr,roc_auc,modname,optimal_thresholdsf))
    Stackmeanscore = (round(np.mean(Stackscores),4),'Stackmeanscore')
    Stackmeanscore_th = (round(np.mean(Stackscores_th),4),'Stackmeanscore_th')
    Stackrocs_sorted = sorted(Stackrocs,key=lambda x:x[2],reverse=True)
    Stackmodname = Stackrocs_sorted[0][3]
    optimal_thresholdsf = Stackrocs_sorted[0][4]
    Stackmodel = Stackmodels[Stackmodname]['Stack']
    WHO_trainpred = Stackmodel.predict_proba(select_U_data_mod)[:,1]
    fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    roc_auch = auc(fprh,tprh)
    if roc_auch < 0.5:
        WHO_trainpred = Stackmodel.predict_proba(select_U_data_mod)[:,0]
        fprh,tprh,thresholdsh = roc_curve(WHO_train,WHO_trainpred)
    optimal_idx = np.argmax(tprh-fprh)
    optimal_thresholdsh = thresholdsh[optimal_idx]
    Stackmean_tpr = np.mean(Stacktprs, axis=0)
    Stackmean_tpr[-1] = 1.0
    Stackmean_auc = auc(Stackmean_fpr, Stackmean_tpr)
    Stackstd_auc = np.std(Stackaucs,ddof=1)
    Stackstd_tpr = np.std(Stacktprs, axis=0)
    Stacktprs_upper = np.minimum(Stackmean_tpr + Stackstd_tpr, 1)
    Stacktprs_lower = np.maximum(Stackmean_tpr - Stackstd_tpr, 0)
    Stackrocresults = {'Stackmodels':Stackmodels,'Stackrocs':Stackrocs_sorted, 'Stackmean_tpr':Stackmean_tpr, 'Stackmean_fpr':Stackmean_fpr, 'Stacktprs_upper':Stacktprs_upper, 'Stacktprs_lower':Stacktprs_lower,'optimal_thresholdsh':optimal_thresholdsh,'optimal_thresholdsf':optimal_thresholdsf,'Stackmeanscore':Stackmeanscore,'Stackmeanscore_th':Stackmeanscore_th}
    allmodrocresults = {'LRrocresults':LRrocresults, 'SVMrocresults':SVMrocresults, 'KNNrocresults':KNNrocresults, 'NBrocresults':NBrocresults, 'RFrocresults':RFrocresults, 'Stackrocresults':Stackrocresults}
    allmodmeanaucs.append((Stackmean_auc,'Stack'))
    print("Stackmodels done.")
    allmodmeanaucs_sorted = sorted(allmodmeanaucs, reverse=True)
    selectmodelname = allmodmeanaucs_sorted[0][1] #根据auc的平均值选出大的模型名称
     #在大的模型名称里面选出roc最高的一折
    for mod in allmodrocresults.keys():
        if selectmodelname in mod:
            selectmodrocresults = allmodrocresults[mod]
            KEYS = list(selectmodrocresults.keys())
            selectmodrocs = selectmodrocresults[KEYS[1]]
            modname = selectmodrocs[0][3]
            selectmodels = selectmodrocresults[KEYS[0]]
            optimal_thresholdsh = selectmodrocresults[KEYS[6]]
            optimal_thresholdsf = selectmodrocresults[KEYS[7]]
            try:
                trainCVacc = selectmodrocresults[KEYS[8]][0]
                trainCVacc_th = selectmodrocresults[KEYS[9]][0]
            except Exception as e:
                trainCVacc = selectmodrocresults[KEYS[6]][0]
                trainCVacc_th = selectmodrocresults[KEYS[7]][0]
            selectmodel = selectmodels[modname]
            selectmodelname = list(selectmodel.keys())[0]
            selectmod = selectmodel[selectmodelname]
            print(selectmodelname)
            try:
                testacc = selectmod.score(select_U_data_test_mod, WHO_test)
                WHO_pred2 = []
                WHO_pred = selectmod.predict_proba(select_U_data_test_mod)[:,1]
                fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
                roc_auc = auc(fpr, tpr)
                if roc_auc < 0.5:
                    WHO_pred = selectmod.predict_proba(select_U_data_test_mod)[:,0]
                for prob in WHO_pred:
                    if prob >= optimal_thresholdsh:
                        WHO_pred2.append(1)
                    else:
                        WHO_pred2.append(0)
                testacc_thrh = accuracy_score(WHO_test, WHO_pred2)
                WHO_pred2f = []
                for prob in WHO_pred:
                    if prob >= optimal_thresholdsf:
                        WHO_pred2f.append(1)
                    else:
                        WHO_pred2f.append(0)
                testacc_thrf = accuracy_score(WHO_test, WHO_pred2f)
            except Exception as e:
                params = selectmod.get_params()
                print(params)
                X_train = selectmodel[list(selectmodel.keys())[1]]
                Y_train = selectmodel[list(selectmodel.keys())[3]]
                try:
                    selectmod = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'],decision_function_shape='ovo', random_state=0).fit(X_train, Y_train)
                except Exception as e:
                    selectmod = StackingClassifier(estimators=params['estimators'],final_estimator=params['final_estimator'],cv=params['cv'],stack_method=params['stack_method'],n_jobs=-1).fit(X_train,Y_train)
                testacc = selectmod.score(select_U_data_test_mod, WHO_test)
                testacc_thrh = testacc
                testacc_thrf = testacc
            allgroups_testaccs2.append((testacc,featurename,modname))
            allgroups_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            featurenum = len(select_U_features_mod)
            if featurenum == 2:
                groups2_testaccs2.append((testacc,featurename,modname))
                groups2_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 3:
                groups3_testaccs2.append((testacc, featurename, modname))
                groups3_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 4:
                groups4_testaccs2.append((testacc,featurename,modname))
                groups4_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 5:
                groups5_testaccs2.append((testacc,featurename,modname))
                groups5_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 6:
                groups6_testaccs2.append((testacc,featurename,modname))
                groups6_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 7:
                groups7_testaccs2.append((testacc,featurename,modname))
                groups7_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 8:
                groups8_testaccs2.append((testacc,featurename,modname))
                groups8_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc, 'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf,'selectmodel':selectmod}
            elif featurenum == 9:
                groups9_testaccs2.append((testacc,featurename,modname))
                groups9_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'trainCVacc_th':trainCVacc_th,'testacc': testacc,'testacc_thrh':testacc_thrh,'testacc_thrf':testacc_thrf, 'selectmodel':selectmod}

    print("\n"+simplefeaturename + 'done.')
allgroups_testaccs_sorted2 = sorted(allgroups_testaccs2, key=lambda x:x[0], reverse=True)
groups2_testaccs_sorted2 = sorted(groups2_testaccs2, key=lambda  x:x[0], reverse=True)
groups3_testaccs_sorted2 = sorted(groups3_testaccs2, key=lambda x:x[0], reverse=True)
groups4_testaccs_sorted2 = sorted(groups4_testaccs2, key=lambda  x:x[0], reverse=True)
groups5_testaccs_sorted2 = sorted(groups5_testaccs2, key=lambda  x:x[0], reverse=True)
groups6_testaccs_sorted2 = sorted(groups6_testaccs2, key=lambda  x:x[0], reverse=True)
groups7_testaccs_sorted2 = sorted(groups7_testaccs2, key=lambda  x:x[0], reverse=True)
groups8_testaccs_sorted2 = sorted(groups8_testaccs2, key=lambda  x:x[0], reverse=True)
groups9_testaccs_sorted2 = sorted(groups9_testaccs2, key=lambda  x:x[0], reverse=True)

groups_testaccs_sorted2 = {'allgroups_testaccs':allgroups_testaccs_sorted2, 'groups2_testaccs':groups2_testaccs_sorted2, 'groups3_testaccs':groups3_testaccs_sorted2, 'groups4_testaccs':groups4_testaccs_sorted2, 'groups5_testaccs':groups5_testaccs_sorted2, 'groups6_testaccs':groups6_testaccs_sorted2, 'groups7_testaccs':groups7_testaccs_sorted2, 'groups8_testaccs':groups8_testaccs_sorted2, 'groups9_testaccs':groups9_testaccs_sorted2}
groups_testresults2 = {'allgroups_testresults':allgroups_testresults2, 'groups2_testresults':groups2_testresults2, 'groups3_testresults':groups3_testresults2, 'groups4_testresults':groups4_testresults2, 'groups5_testresults':groups5_testresults2, 'groups6_testresults':groups6_testresults2, 'groups7_testresults':groups7_testresults2, 'groups8_testresults':groups8_testresults2, 'groups9_testresults':groups9_testresults2}
groups_test2 = {'groups_testaccs_sorted':groups_testaccs_sorted2, 'groups_testrestuls2':groups_testresults2}
joblib.dump(groups_test2, "./estimatorprediction/groups_rocresults_radiomics.pkl")
print("\n AUC ALLgroups done.")

#画ROC曲线，计算AUC，最佳截止值处的最佳特异性和最佳敏感性--------------1、5-5fold 交叉验证计算平均的ROC曲线及AUC值，然后选择大的模型；2、在大的模型名称下选择5-5fold中AUC值最大的那个模型作为最终模型进行训练

#groupsaucbest_accs_innertest = joblib.load("/home/xujq/brain/DWImodels/adults/adults_dwiresults/estimatorprediction/foldbest2_1accs.pkl")
"""
groups_test2 = joblib.load("/home2/HWGroup/xujq/brain/DWImodels/renji/b_4500/dataanalysis/groups_rocresults.pkl")
groups_testresults2 = groups_test2['groups_testrestuls2']
groups_testaccs_sorted2 = groups_test2['groups_testaccs_sorted']
select_features = joblib.load("/home/xujq/brain/DWImodels/adults/adults_dwiresults/estimatorprediction/select_features.pkl")
datapreparing = joblib.load('/home2/HWGroup/xujq/brain/DWImodels/renji/b_4500/dataanalysis/datapreparing_renji.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
U_featurenames = traindataall['U_featurenames'].tolist()
U_data = traindataall['U_data']
U_data_test = testdataall['U_data_test']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
featurenames = traindataall['featurenames'].tolist()
data_test = testdataall['data']
print("AUC Load prepared data done.")
"""
#主要是想利用一下inner testset上的optimal——threshold去external testset上看一下诊断的准确率如何
groupsaucbest_accs = {}
for groupresultsname in groups_testresults2:
    plt.cla()
    simplegroupname = groupresultsname.replace('_testresults','')
    groupaccsname = groupresultsname.replace('results','accs')
    n_groups_testresults2 = groups_testresults2[groupresultsname]
    n_groups_testaccs_sorted2 = groups_testaccs_sorted2[groupaccsname]

    bestauc_accs = []
    for tup in n_groups_testaccs_sorted2:
        testacc_original = tup[0]
        featurename = tup[1]
        bestmodelname = tup[2]
        trainCVacc = n_groups_testresults2[featurename]['trainCVacc']
        trainCVacc_th = n_groups_testresults2[featurename]['trainCVacc_th']
        testacc_thrh = n_groups_testresults2[featurename]['testacc_thrh']
        testacc_thrf = n_groups_testresults2[featurename]['testacc_thrf']
        trainCVauc = n_groups_testresults2[featurename]['allmodmeanaucs'][0][0]
        bestmodel = n_groups_testresults2[featurename]['selectmodel']
        Select_U_data_test_mod = n_groups_testresults2[featurename]['select_U_data_test_mod']
        if 'SVM' in bestmodelname:
            WHO_pred = bestmodel.predict(Select_U_data_test_mod)
        else:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,1]
        simplefeaturename = featurename.replace('select_features_','')
        fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
        roc_auc = auc(fpr,tpr)
        if (roc_auc < 0.5)&(~('SVM' in bestmodelname)):
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,0]
            fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
            roc_auc = auc(fpr, tpr)
        if simplefeaturename == 'allmod':
            fprallmod = fpr
            tprallmod = tpr
            roc_aucallmod = roc_auc
            bestmodelnameallmod = bestmodelname
        plt.plot(fpr, tpr, lw=1, label = 'ROC %s-%s (AUC = %0.2f)'%(simplefeaturename,bestmodelname,roc_auc))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_tpr = tpr[optimal_idx]
        best_tnr = 1 - fpr[optimal_idx]
        if 'SVM' in bestmodelname:
            testacc_thr = testacc_original
            testacc_thrh = testacc_original
            testacc_thrf = testacc_original

        bestauc_accs.append((simplefeaturename,bestmodelname,round(trainCVacc,4),round(trainCVacc_th,4),round(trainCVauc,4),round(testacc_original,4),round(testacc_thrh,4),round(testacc_thrf,4),round(roc_auc,4),round(best_tpr,4),round(best_tnr,4),round(optimal_threshold,4)))
    plt.plot([0,1], [0,1], color='red', lw=1, alpha = 0.3,linestyle = '--', label = 'Reference Line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(simplegroupname+' test ROC (radiomics)')
    plt.legend(loc = "lower right", fontsize = 7)
    if simplegroupname == 'allgroups':
        plt.legend(loc="lower right", fontsize=5)
    plt.savefig("./estimatorprediction/"+simplegroupname+"testROC_radiomics.png",dpi=300)
    plt.show()
    groupsaucbest_accs[simplegroupname] = bestauc_accs
    print(simplegroupname + "done.")
joblib.dump(groupsaucbest_accs, "./estimatorprediction/foldbest2_1accs_radiomics.pkl")

# 把10个特征一起test的曲线单独画出来
plt.cla()
plt.plot(fprallmod, tprallmod, lw=2, label = 'ROC allmod-%s (AUC = %0.2f)'%(bestmodelnameallmod,roc_aucallmod))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('allmod test ROC (radiomics)')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./estimatorprediction/allmod_testROC_radiomics.png",dpi=300)
plt.show()


print("\n roc 2-1 done.")

