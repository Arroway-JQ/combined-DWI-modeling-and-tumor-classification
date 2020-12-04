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
#%% ------------featuregroup 的最终在testdataset上的结果（用平均AUC选模）
ALLgroups = joblib.load("./ALLgroups.pkl")
ALLgroup_bestmodscore = ALLgroups['ALLgroup_bestmodscore']
ALLgroup_bestmodmeanscore_sorted = ALLgroups['ALLgroup_bestmodmeanscore']
ALLgroupresults = ALLgroups['ALLgroupresults']
select_features = joblib.load("./select_features.pkl")
datapreparing = joblib.load('./datapreparing.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
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

    for k in LRmodels.keys():
        modname = k
        modelresults = LRmodels[k]
        LRCV = modelresults['LRCV']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        Y_pred = LRCV.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        LRtprs.append(np.interp(LRmean_fpr, fpr, tpr))
        LRtprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        LRaucs.append(roc_auc)
        LRrocs.append((fpr,tpr,roc_auc,modname))
    LRrocs_sorted = sorted(LRrocs,key=lambda x:x[2])
    LRmean_tpr = np.mean(LRtprs, axis=0)
    LRmean_tpr[-1] = 1.0
    LRmean_auc = auc(LRmean_fpr, LRmean_tpr)
    LRstd_auc = np.std(LRaucs,ddof=1)
    LRstd_tpr = np.std(LRtprs, axis=0)
    LRtprs_upper = np.minimum(LRmean_tpr + LRstd_tpr, 1)
    LRtprs_lower = np.maximum(LRmean_tpr - LRstd_tpr, 0)
    LRrocresults = {'LRmodels':LRmodels, 'LRrocs':LRrocs_sorted, 'LRmean_tpr':LRmean_tpr, 'LRmean_fpr':LRmean_fpr, 'LRtprs_upper':LRtprs_upper, 'LRtprs_lower':LRtprs_lower}
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
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        SVMtprs.append(np.interp(SVMmean_fpr, fpr, tpr))
        SVMtprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        SVMaucs.append(roc_auc)
        SVMrocs.append((fpr,tpr,roc_auc,modname))
    SVMrocs_sorted = sorted(SVMrocs,key=lambda x:x[2])  #按照roc_auc来进行排序
    SVMmean_tpr = np.mean(SVMtprs, axis=0)
    SVMmean_tpr[-1] = 1.0
    SVMmean_auc = auc(SVMmean_fpr, SVMmean_tpr)
    SVMstd_auc = np.std(SVMaucs,ddof=1)
    SVMstd_tpr = np.std(SVMtprs, axis=0)
    SVMtprs_upper = np.minimum(SVMmean_tpr + SVMstd_tpr, 1)
    SVMtprs_lower = np.maximum(SVMmean_tpr - SVMstd_tpr, 0)
    SVMrocresults = {'SVMmodels':SVMmodels,'SVMrocs':SVMrocs_sorted, 'SVMmean_tpr':SVMmean_tpr, 'SVMmean_fpr':SVMmean_fpr, 'SVMtprs_upper':SVMtprs_upper, 'SVMtprs_lower':SVMtprs_lower}
    allmodmeanaucs.append((SVMmean_auc,'SVM'))
    print("SVMmodels done.")
    for k in KNNmodels.keys():
        modname = k
        modelresults = KNNmodels[k]
        KNNCV = modelresults['KNNCV']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        Y_pred = KNNCV.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        KNNtprs.append(np.interp(KNNmean_fpr, fpr, tpr))
        KNNtprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        KNNaucs.append(roc_auc)
        KNNrocs.append((fpr,tpr,roc_auc,modname))
    KNNrocs_sorted = sorted(KNNrocs,key=lambda x:x[2])
    KNNmean_tpr = np.mean(KNNtprs, axis=0)
    KNNmean_tpr[-1] = 1.0
    KNNmean_auc = auc(KNNmean_fpr, KNNmean_tpr)
    KNNstd_auc = np.std(KNNaucs,ddof=1)
    KNNstd_tpr = np.std(KNNtprs, axis=0)
    KNNtprs_upper = np.minimum(KNNmean_tpr + KNNstd_tpr, 1)
    KNNtprs_lower = np.maximum(KNNmean_tpr -KNNstd_tpr, 0)
    KNNrocresults = {'KNNmodels':KNNmodels, 'KNNrocs':KNNrocs_sorted, 'KNNmean_tpr':KNNmean_tpr, 'KNNmean_fpr':KNNmean_fpr, 'KNNtprs_upper':KNNtprs_upper, 'KNNtprs_lower':KNNtprs_lower}
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
        NBtprs.append(np.interp(NBmean_fpr, fpr, tpr))
        NBtprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        NBaucs.append(roc_auc)
        NBrocs.append((fpr,tpr,roc_auc,modname))
    NBrocs_sorted = sorted(NBrocs,key=lambda x:x[2])
    NBmean_tpr = np.mean(NBtprs, axis=0)
    NBmean_tpr[-1] = 1.0
    NBmean_auc = auc(NBmean_fpr, NBmean_tpr)
    NBstd_auc = np.std(NBaucs,ddof=1)
    NBstd_tpr = np.std(NBtprs, axis=0)
    NBtprs_upper = np.minimum(NBmean_tpr + NBstd_tpr, 1)
    NBtprs_lower = np.maximum(NBmean_tpr -NBstd_tpr, 0)
    NBrocresults = {'NBmodels':NBmodels, 'NBrocs':NBrocs_sorted, 'NBmean_tpr':NBmean_tpr, 'NBmean_fpr':NBmean_fpr, 'NBtprs_upper':NBtprs_upper, 'NBtprs_lower':NBtprs_lower}
    allmodmeanaucs.append((NBmean_auc,'NB'))
    print("NB models done.")
    for k in RFmodels.keys():
        modname = k
        modelresults = RFmodels[k]
        RF = modelresults['RF']
        X_test = modelresults['X_test']
        Y_test = modelresults['Y_test']
        Y_pred = RF.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        RFtprs.append(np.interp(RFmean_fpr, fpr, tpr))
        RFtprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        RFaucs.append(roc_auc)
        RFrocs.append((fpr,tpr,roc_auc,modname))
    RFrocs_sorted = sorted(RFrocs,key=lambda x:x[2])
    RFmean_tpr = np.mean(RFtprs, axis=0)
    RFmean_tpr[-1] = 1.0
    RFmean_auc = auc(RFmean_fpr, RFmean_tpr)
    RFstd_auc = np.std(RFaucs,ddof=1)
    RFstd_tpr = np.std(RFtprs, axis=0)
    RFtprs_upper = np.minimum(RFmean_tpr + RFstd_tpr, 1)
    RFtprs_lower = np.maximum(RFmean_tpr - RFstd_tpr, 0)
    RFrocresults = {'RFmodels':RFmodels,'RFrocs':RFrocs_sorted, 'RFmean_tpr':RFmean_tpr, 'RFmean_fpr':RFmean_fpr, 'RFtprs_upper':RFtprs_upper, 'RFtprs_lower':RFtprs_lower}
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
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        Stacktprs.append(np.interp(Stackmean_fpr, fpr, tpr))
        Stacktprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        Stackaucs.append(roc_auc)
        Stackrocs.append((fpr,tpr,roc_auc,modname))
    Stackrocs_sorted = sorted(Stackrocs,key=lambda x:x[2],reverse=True)
    Stackmean_tpr = np.mean(Stacktprs, axis=0)
    Stackmean_tpr[-1] = 1.0
    Stackmean_auc = auc(Stackmean_fpr, Stackmean_tpr)
    Stackstd_auc = np.std(Stackaucs,ddof=1)
    Stackstd_tpr = np.std(Stacktprs, axis=0)
    Stacktprs_upper = np.minimum(Stackmean_tpr + Stackstd_tpr, 1)
    Stacktprs_lower = np.maximum(Stackmean_tpr - Stackstd_tpr, 0)
    Stackrocresults = {'Stackmodels':Stackmodels,'Stackrocs':Stackrocs_sorted, 'Stackmean_tpr':Stackmean_tpr, 'Stackmean_fpr':Stackmean_fpr, 'Stacktprs_upper':Stacktprs_upper, 'Stacktprs_lower':Stacktprs_lower}
    allmodrocresults = {'LRrocresults':LRrocresults, 'SVMrocresults':SVMrocresults, 'KNNrocresults':KNNrocresults, 'NBrocresults':NBrocresults, 'RFrocresults':RFrocresults, 'Stackrocresults':Stackrocresults}
    allmodmeanaucs.append((Stackmean_auc,'Stack'))
    print("Stackmodels done.")
    allmodmeanaucs_sorted = sorted(allmodmeanaucs, reverse=True)
    selectmodelname = allmodmeanaucs_sorted[0][1] #根据auc的平均值选出大的模型名称
    for tup in ALLmodmeanscores_sorted: #在大的模型名称里面选出roc最高的一折
        if selectmodelname in tup[1]:
            trainCVacc = tup[0]
    for mod in allmodrocresults.keys():
        if selectmodelname in mod:
            selectmodrocresults = allmodrocresults[mod]
            KEYS = list(selectmodrocresults.keys())
            selectmodrocs = selectmodrocresults[KEYS[1]]
            modname = selectmodrocs[0][3]
            selectmodels = selectmodrocresults[KEYS[0]]
            selectmodel = selectmodels[modname]
            selectmodelname = list(selectmodel.keys())[0]
            selectmod = selectmodel[selectmodelname]
            try:
                testacc = selectmod.score(select_U_data_test_mod, WHO_test)
            except Exception as e:
                params = selectmod.get_params()
                X_train = selectmodel[list(selectmodel.keys())[1]]
                Y_train = selectmodel[list(selectmodel.keys())[3]]
                try:
                    selectmod = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'],decision_function_shape='ovo', random_state=0).fit(X_train, Y_train)
                except Exception as e:
                    selectmod = StackingClassifier(estimators=params['estimators'],final_estimator=params['final_estimator'],cv=params['cv'],stack_method=params['stack_method'],n_jobs=-1).fit(X_train,Y_train)
                testacc = selectmod.score(select_U_data_test_mod, WHO_test)
            allgroups_testaccs2.append((testacc,featurename,modname))
            allgroups_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            featurenum = len(select_U_features_mod)
            if featurenum == 2:
                groups2_testaccs2.append((testacc,featurename,modname))
                groups2_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 3:
                groups3_testaccs2.append((testacc, featurename, modname))
                groups3_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 4:
                groups4_testaccs2.append((testacc,featurename,modname))
                groups4_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 5:
                groups5_testaccs2.append((testacc,featurename,modname))
                groups5_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 6:
                groups6_testaccs2.append((testacc,featurename,modname))
                groups6_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 7:
                groups7_testaccs2.append((testacc,featurename,modname))
                groups7_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 8:
                groups8_testaccs2.append((testacc,featurename,modname))
                groups8_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}
            elif featurenum == 9:
                groups9_testaccs2.append((testacc,featurename,modname))
                groups9_testresults2[key] = {'allmodrocresults': allmodrocresults,'allmodmeanaucs': allmodmeanaucs_sorted,'select_U_features_mod': select_features_mod,'select_U_data_mod': select_U_data_mod,'select_U_data_test_mod': select_U_data_test_mod, 'trainCVacc': trainCVacc,'testacc': testacc, 'selectmodel':selectmod}

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
joblib.dump(groups_test2, "./groups_rocresults.pkl")
print("\n AUC ALLgroups done.")

#%%画ROC曲线，计算AUC，最佳截止值处的最佳特异性和最佳敏感性--------------1、5-5fold 交叉验证计算平均的ROC曲线及AUC值，然后选择大的模型；2、在大的模型名称下选择5-5fold中AUC值最大的那个模型作为最终模型进行训练
groupsaucbest_accs = {}
for groupresultsname in groups_testresults2:
    simplegroupname = groupresultsname.replace('_testresults','')
    groupaccsname = groupresultsname.replace('results','accs')
    n_groups_testresults2 = groups_testresults2[groupresultsname]
    n_groups_testaccs_sorted2 = groups_testaccs_sorted2[groupaccsname]

    plt.cla()
    bestauc_accs = []
    for tup in n_groups_testaccs_sorted2:
        testacc_roiginal = tup[0]
        featurename = tup[1]
        bestmodelname = tup[2]
        trainCVacc = n_groups_testresults2[featurename]['trainCVacc']
        trainCVauc = n_groups_testresults2[featurename]['allmodmeanaucs'][0][0]
        bestmodel = n_groups_testresults2[featurename]['selectmodel']
        Select_U_data_test_mod = n_groups_testresults2[featurename]['select_U_data_test_mod']
        if 'SVM' in bestmodelname:
            WHO_pred = bestmodel.predict(Select_U_data_test_mod)
        else:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,1]
        simplefeaturename = featurename.replace('select_features_','')
        if featurename == 'select_featrues_Caver':
            simplefeaturename = featurename.replace('select_featrues_','')
        fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
        roc_auc = auc(fpr,tpr)
        if roc_auc < 0.5:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,0]
            fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
            roc_auc = auc(fpr, tpr)
        if simplefeaturename == 'allmod':
            fprallmod = fpr
            tprallmod = tpr
            roc_aucallmod = roc_auc
            bestmodelnameallmod = bestmodelname
        elif simplefeaturename == 'Aaver':
            fprAaver = fpr
            tprAaver = tpr
            roc_aucAaver = roc_auc
            bestmodelnameAaver = bestmodelname
        plt.plot(fpr, tpr, lw=2, alpha = 0.3, label = 'ROC %s-%s (AUC = %0.2f)'%(simplefeaturename,bestmodelname,roc_auc))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_tpr = tpr[optimal_idx]
        best_tnr = 1 - fpr[optimal_idx]
        WHO_pred2 = []
        bestauc_accs.append((simplefeaturename,bestmodelname,round(trainCVacc,4),round(trainCVauc,4),round(testacc_roiginal,4),round(roc_auc,4),round(best_tpr,4),round(best_tnr,4),round(optimal_threshold,4)))
    plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(simplegroupname+'best featuresmodel2-1 test ROC')
    plt.legend(loc = "lower right", fontsize = 7)
    if simplegroupname == 'allgroups':
        plt.legend(loc="lower right", fontsize=5)
    plt.savefig("./"+simplegroupname+"best_featuresmod2-1_testROCs.png")
    plt.show()
    groupsaucbest_accs[simplegroupname] = bestauc_accs
    print(simplegroupname + "done.")
joblib.dump(groupsaucbest_accs, "./foldbest2_1accs.pkl")
# 把10个特征一起test的曲线单独画出来
plt.cla()
plt.plot(fprallmod, tprallmod, lw=2, alpha = 0.3, label = 'ROC allmod-%s (AUC = %0.2f)'%(bestmodelnameallmod,roc_aucallmod))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('allmodbest featuresmodel2-1 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./allmodbest_featuresmod2-1_testROCs.png")
plt.show()
#把ADCaver的曲线单独画出来
plt.cla()
plt.plot(fprAaver, tprAaver, lw=2, alpha = 0.3, label = 'ROC Aaver-%s (AUC = %0.2f)'%(bestmodelnameAaver,roc_aucAaver))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aaverbest featuresmodel2-1 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./Aaverbest_featuresmod2-1_testROCs.png")
plt.show()
print("\n roc 2-1 done.")

#%%#%% 画ROC曲线，计算AUC，最佳截止值处的最佳特异性和最佳敏感性--------------选模过程：1、5-5fold交叉验证计算平均auc找到最好的大的模型；2、获取训练过的模型的参数在整体trainset上训练得到最终的模型
groups_test2 = joblib.load("./groups_rocresults.pkl")
groups_testresults2 = groups_test2['groups_testrestuls2']
allgroups_testresults2 = groups_testresults2['allgroups_testresults']
groups_testaccs_sorted2 = groups_test2['groups_testaccs_sorted']
allgroups_testaccs2 = groups_testaccs_sorted2['allgroups_testaccs']
groupsrocbest_meanmod_accs = {}
for groupresultsname in groups_testresults2:
    simplegroupname = groupresultsname.replace('_testresults','')
    groupaccsname = groupresultsname.replace('results','accs')
    n_groups_testresults2 = groups_testresults2[groupresultsname]
    n_groups_testaccs_sorted2 = groups_testaccs_sorted2[groupaccsname]

    plt.cla()
    bestauc_meanmod_accs=[]
    for tup in n_groups_testaccs_sorted2:
        featurename = tup[1]
        bestmodelname = tup[2]
        trainCVacc = allgroups_testresults2[featurename]['trainCVacc']
        trainCVauc = allgroups_testresults2[featurename]['allmodmeanaucs'][0][0]
        bestmodel = allgroups_testresults2[featurename]['selectmodel']
        Select_U_data_test_mod = allgroups_testresults2[featurename]['select_U_data_test_mod']
        Select_U_data_mod = allgroups_testresults2[featurename]['select_U_data_mod']
        if 'LR' in bestmodelname:
            params = bestmodel.get_params()
            bestmodel = LogisticRegressionCV(Cs=params['Cs'], cv=params['cv'], n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        if 'SVM' in bestmodelname:
            params = bestmodel.get_params()
            bestmodel = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'],decision_function_shape='ovo', random_state=0).fit(Select_U_data_mod, WHO_train)
        if 'KNN' in bestmodelname:
            params = bestmodel.get_params()
            bestmodel = KNeighborsClassifier(n_neighbors=params['n_neighbors'], algorithm=params['algorithm'],leaf_size=params['leaf_size'], metric='minkowski', n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        if 'NB' in bestmodelname:
            bestmodel = GaussianNB().fit(Select_U_data_mod, WHO_train)
        if 'RF' in bestmodelname:
            params = bestmodel.get_params()
            bestmodel = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],min_samples_split=params['min_samples_split'],min_samples_leaf=params['min_samples_leaf'],max_features=params['max_features'], oob_score=True, n_jobs=-1,random_state=0).fit(Select_U_data_mod, WHO_train)
        if 'Stack' in bestmodelname:
            params = bestmodel.get_params()
            bestmodel = StackingClassifier(estimators=params['estimators'], final_estimator=params['final_estimator'],stack_method='auto', n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        if bestmodelname == 'SVMCV':
            WHO_pred = bestmodel.predict(Select_U_data_test_mod)
        else:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:, 1]
        testacc_original = bestmodel.score(Select_U_data_test_mod, WHO_test)
        simplefeaturename = featurename.replace('select_features_', '')
        if featurename == 'select_featrues_Caver':
            simplefeaturename = featurename.replace("select_featrues_","")
        fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
        roc_auc = auc(fpr, tpr)
        if roc_auc < 0.5:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,0]
            fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
            roc_auc = auc(fpr, tpr)
        if simplefeaturename == 'allmod':
            fprallmod = fpr
            tprallmod = tpr
            roc_aucallmod = roc_auc
            bestmodelnameallmod = bestmodelname
        elif simplefeaturename == 'Aaver':
            fprAaver = fpr
            tprAaver = tpr
            roc_aucAaver = roc_auc
            bestmodelnameAaver = bestmodelname
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC %s-%s (AUC = %0.2f)' % (simplefeaturename, bestmodelname, roc_auc))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_tpr = tpr[optimal_idx]
        best_tnr = 1 - fpr[optimal_idx]
        bestauc_meanmod_accs.append((simplefeaturename, bestmodelname, round(trainCVacc, 4), round(trainCVauc,4), round(testacc_original, 4), round(roc_auc, 4), round(best_tpr, 4), round(best_tnr, 4), round(optimal_threshold, 4)))
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Reference Line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(simplegroupname+'best featuresmodel2-2 test ROC')
    plt.legend(loc='lower right', fontsize=7)
    if simplegroupname == 'allgroups':
        plt.legend(loc="lower right", fontsize=5)
    plt.savefig("./"+simplegroupname+"best_featuresmod2-2_testROCs.png")
    plt.show()
    groupsrocbest_meanmod_accs[simplegroupname] = bestauc_meanmod_accs
    print(simplegroupname + "done.")
joblib.dump(groupsrocbest_meanmod_accs, "./best_meanmod2-2_accs.pkl")
# 把10个特征一起test的曲线单独画出来
plt.cla()
plt.plot(fprallmod, tprallmod, lw=2, alpha = 0.3, label = 'ROC allmod-%s (AUC = %0.2f)'%(bestmodelnameallmod,roc_aucallmod))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('allmodbest featuresmodel2-2 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./allmodbest_featuresmod2-2_testROCs.png")
plt.show()
#把ADCaver的曲线单独画出来
plt.cla()
plt.plot(fprAaver, tprAaver, lw=2, alpha = 0.3, label = 'ROC Aaver-%s (AUC = %0.2f)'%(bestmodelnameAaver,roc_aucAaver))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aaverbest featuresmodel2-2 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./Aaverbest_featuresmod2-2_testROCs.png")
plt.show()
print("\n roc 2-2 done.")

#%%-----------featuregroup 的最终在testdataset上的结果（根据平均准确率选模）
ALLgroups = joblib.load("./ALLgroups.pkl")
ALLgroup_bestmodscore = ALLgroups['ALLgroup_bestmodscore']
ALLgroup_bestmodmeanscore_sorted = ALLgroups['ALLgroup_bestmodmeanscore']
ALLgroupresults = ALLgroups['ALLgroupresults']
groups_test2 = joblib.load("./groups_rocresults.pkl")
groups_testresults2 = groups_test2['groups_testrestuls2']
allgroups_testresults2 = groups_testresults2['allgroups_testresults']
groups_testaccs_sorted2 = groups_test2['groups_testaccs_sorted']
allgroups_testaccs2 = groups_testaccs_sorted2['allgroups_testaccs']
select_features = joblib.load("./select_features.pkl")
datapreparing = joblib.load('./datapreparing.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
print("ACC Load prepared data done.")

allgroups_testaccs = []
allgroups_testresults = {}
groups2_testaccs = [] #数字2 表示在这个组合中一共使用了2个feature
groups2_testresults = {}
groups3_testaccs = []
groups3_testresults = {}
groups4_testaccs = []
groups4_testresults = {}
groups5_testaccs = []
groups5_testresults = {}
groups6_testaccs = []
groups6_testresults = {}
groups7_testaccs = []
groups7_testresults = {}
groups8_testaccs = []
groups8_testresults = {}
groups9_testaccs = []
groups9_testresults = {}
for tup in ALLgroup_bestmodmeanscore_sorted:
    key = tup[1]
    groupresults = ALLgroupresults[key]
    groupresults2 = allgroups_testresults2[key]
    allmodmeanaucs = groupresults2['allmodmeanaucs']
    ALLmod = groupresults['ALLmod']
    ALLmodscores_sorted = groupresults['ALLmodscores']
    ALLmodmeanscores_sorted = groupresults['ALLmodmeanscores']
    selectmodelname = ALLmodmeanscores_sorted[0][1]
    selectmodelname = selectmodelname.replace('meanscore','')
    for Tup in allmodmeanaucs:
        tupname = Tup[1]
        if selectmodelname in tupname:
            trainCVauc = Tup[0]
    for tup2 in ALLmodscores_sorted:
        modname = tup2[1]
        if selectmodelname in modname:
            selectmodelname = modname
            break
    selectmodelresults = ALLmod[selectmodelname]
    selectmodelname = list(selectmodelresults.keys())[0]
    selectmodel = selectmodelresults[selectmodelname]
    trainCVacc = ALLmodmeanscores_sorted[0][0]
    val = select_features[key]
    select_features_mod = val
    select_U_features_mod = list(val.values())[0]
    select_U_data_mod = list(val.values())[1]
    select_U_data_test_mod = list(val.values())[2]
    try:
        testacc = selectmodel.score(select_U_data_test_mod, WHO_test)
    except Exception as e:
        params = selectmodel.get_params()
        X_train = list(selectmodelresults.values())[1]
        Y_train = list(selectmodelresults.values())[3]
        try:
            selectmodel = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'],decision_function_shape='ovo', random_state=0).fit(X_train, Y_train)
        except Exception as e:
            selectmodel = StackingClassifier(estimators=params['estimators'],final_estimator=params['final_estimator'],cv=params['cv'], stack_method=params['stack_method'],n_jobs=-1).fit(X_train,Y_train)
        testacc = selectmodel.score(select_U_data_test_mod, WHO_test)
    testresults = {'trainCVacc':trainCVacc, 'trainCVauc':trainCVauc,'testacc':testacc, 'select_features_mod':val, 'selectmodelname':selectmodelname, 'selectmodel':selectmodel}
    allgroups_testaccs.append((testacc, key))
    allgroups_testresults[key] = testresults
    featurenum = len(select_U_features_mod)
    if featurenum == 2:
        groups2_testaccs.append((testacc, key))
        groups2_testresults[key] = testresults
    elif featurenum == 3:
        groups3_testaccs.append((testacc, key))
        groups3_testresults[key] = testresults
    elif featurenum == 4:
        groups4_testaccs.append((testacc, key))
        groups4_testresults[key] = testresults
    elif featurenum == 5:
        groups5_testaccs.append((testacc, key))
        groups5_testresults[key] = testresults
    elif featurenum == 6:
        groups6_testaccs.append((testacc, key))
        groups6_testresults[key] = testresults
    elif featurenum == 7:
        groups7_testaccs.append((testacc, key))
        groups7_testresults[key] = testresults
    elif featurenum == 8:
        groups8_testaccs.append((testacc, key))
        groups8_testresults[key] = testresults
    elif featurenum == 9:
        groups9_testaccs.append((testacc, key))
        groups9_testresults[key] = testresults
    print(key.replace('select_U_features_','')+"done.")
allgroups_testaccs_sorted = sorted(allgroups_testaccs, reverse=True)
groups2_testaccs_sorted = sorted(groups2_testaccs, reverse=True)
groups3_testaccs_sorted = sorted(groups3_testaccs, reverse=True)
groups4_testaccs_sorted = sorted(groups4_testaccs, reverse=True)
groups5_testaccs_sorted = sorted(groups5_testaccs, reverse=True)
groups6_testaccs_sorted = sorted(groups6_testaccs, reverse=True)
groups7_testaccs_sorted = sorted(groups7_testaccs, reverse=True)
groups8_testaccs_sorted = sorted(groups8_testaccs, reverse=True)
groups9_testaccs_sorted = sorted(groups9_testaccs, reverse=True)

groups_testaccs_sorted = {'allgroups_testaccs':allgroups_testaccs_sorted, 'groups2_testaccs':groups2_testaccs_sorted, 'groups3_testaccs':groups3_testaccs_sorted,'groups4_testaccs':groups4_testaccs_sorted, 'groups5_testaccs':groups5_testaccs_sorted, 'groups6_testaccs':groups6_testaccs_sorted, 'groups7_testaccs':groups7_testaccs_sorted, 'groups8_testaccs':groups8_testaccs_sorted, 'groups9_testaccs':groups9_testaccs_sorted}
groups_testresults = {'allgroups_testresults':allgroups_testresults, 'groups2_testresults':groups2_testresults, 'groups3_testresults':groups3_testresults, 'groups4_testresults':groups4_testresults, 'groups5_testresults':groups5_testresults, 'groups6_testresults':groups6_testresults, 'groups7_testresults':groups7_testresults, 'groups8_testresults':groups8_testresults, 'groups9_testresults':groups9_testresults}
groups_test = {'groups_testaccs_sorted':groups_testaccs_sorted, 'groups_testresults':groups_testresults}

joblib.dump(groups_test, "./groups_results.pkl")
print("\n ACC ALLgroups done.")

# 画ROC曲线，计算AUC，最佳截止值处的最佳特异性和最佳敏感性-----------选模过程：1、利用5-5fold交叉验证的平均准确率选出大的模型名称；2、在大的模型名称中选出5-5fold中表现最好的那一折的模型进行预测
"""groups_test = joblib.load("./groups_results.pkl")
groups_testresults = groups_test['groups_testresults']
groups_testaccs_sorted = groups_test['groups_testaccs_sorted']
allgroups_testresults = groups_testresults['allgroups_testresults']
select_features = joblib.load("./select_features.pkl")
datapreparing = joblib.load('./datapreparing.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
"""
groupsbest_accs = {}
for groupresultsname in groups_testresults:
    simplegroupname = groupresultsname.replace('_testresults','')
    groupaccsname = groupresultsname.replace('results','accs')
    n_groups_testresults = groups_testresults[groupresultsname]
    n_groups_testaccs_sorted = groups_testaccs_sorted[groupaccsname]

    plt.cla()
    best_accs = []
    for tup in n_groups_testaccs_sorted:
        testacc_original = tup[0]
        featurename = tup[1]
        trainCVauc = allgroups_testresults[featurename]['trainCVauc']
        trainCVacc = allgroups_testresults[featurename]['trainCVacc']
        bestmodelname = allgroups_testresults[featurename]['selectmodelname']
        bestmodel = allgroups_testresults[featurename]['selectmodel']
        key = list(allgroups_testresults[featurename]['select_features_mod'].keys())[2]
        Select_U_data_test_mod = allgroups_testresults[featurename]['select_features_mod'][key]
        if bestmodelname == 'SVMCV':
            WHO_pred = bestmodel.predict(Select_U_data_test_mod)
        else:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,1]
        simplefeaturename = featurename.replace('select_features_','')
        if featurename == 'select_featrues_Caver':
            simplefeaturename = featurename.replace('select_featrues_','')
        fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
        roc_auc = auc(fpr,tpr)
        if (roc_auc < 0.5)&(bestmodelname != 'SVMCV'):
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,0]
            fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
            roc_auc = auc(fpr, tpr)
        if simplefeaturename == 'allmod':
            fprallmod = fpr
            tprallmod = tpr
            roc_aucallmod = roc_auc
            bestmodelnameallmod = bestmodelname
        elif simplefeaturename == 'Aaver':
            fprAaver = fpr
            tprAaver = tpr
            roc_aucAaver = roc_auc
            bestmodelnameAaver = bestmodelname
        plt.plot(fpr, tpr, lw=2, alpha = 0.3, label = 'ROC %s-%s (AUC = %0.2f)'% (simplefeaturename,bestmodelname,roc_auc))
        optimal_idx = np.argmax(tpr -fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_tpr = tpr[optimal_idx]
        best_tnr = 1-fpr[optimal_idx]
        WHO_pred2 = []
        best_accs.append((simplefeaturename,bestmodelname,round(trainCVacc,4),round(trainCVauc,4), round(testacc_original,4),round(roc_auc,4),round(best_tpr,4),round(best_tnr,4),round(optimal_threshold,4)))
    plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(simplegroupname+'best featuresmodel1-1 test ROC')
    plt.legend(loc = "lower right", fontsize = 7)
    if simplegroupname == 'allgroups':
        plt.legend(loc="lower right", fontsize=5)
    plt.savefig("./"+simplegroupname+"best_featuresmod1-1_testROCs.png")
    plt.show()
    groupsbest_accs[simplegroupname] = best_accs
    print(simplegroupname + "done.")
joblib.dump(groupsbest_accs, "./foldbest1_1accs.pkl")
# 把10个特征一起test的曲线单独画出来
plt.cla()
plt.plot(fprallmod, tprallmod, lw=2, alpha = 0.3, label = 'ROC allmod-%s (AUC = %0.2f)'%(bestmodelnameallmod,roc_aucallmod))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('allmodbest featuresmodel1-1 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./allmodbest_featuresmod1-1_testROCs.png")
plt.show()
#把ADCaver的曲线单独画出来
plt.cla()
plt.plot(fprAaver, tprAaver, lw=2, alpha = 0.3, label = 'ROC Aaver-%s (AUC = %0.2f)'%(bestmodelnameAaver,roc_aucAaver))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aaverbest featuresmodel1-1 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./Aaverbest_featuresmod1-1_testROCs.png")
plt.show()

print("\n roc 1-1 done.")

#%% 画ROC曲线，计算AUC，最佳截止值处的最佳特异性和最佳敏感性--------------选模过程：1、5-5fold交叉验证计算平均准确率找到最好的大的模型；2、获取训练过的模型的参数在整体trainset上训练得到最终的模型
groupsbest_meanmod_accs = {}
for groupresultsname in groups_testresults:
    simplegroupname = groupresultsname.replace('_testresults','')
    groupaccsname = groupresultsname.replace('results','accs')
    n_groups_testresults = groups_testresults[groupresultsname]
    n_groups_testaccs_sorted = groups_testaccs_sorted[groupaccsname]

    plt.cla()
    best_meanmod_accs=[]
    for tup in n_groups_testaccs_sorted:
        featurename = tup[1]
        trainCVacc = allgroups_testresults[featurename]['trainCVacc']
        trainCVauc = allgroups_testresults[featurename]['trainCVauc']
        bestmodelname = allgroups_testresults[featurename]['selectmodelname']
        bestmodel = allgroups_testresults[featurename]['selectmodel']
        keys = list(allgroups_testresults[featurename]['select_features_mod'].keys())
        Select_U_data_test_mod = allgroups_testresults[featurename]['select_features_mod'][keys[2]]
        Select_U_data_mod = allgroups_testresults[featurename]['select_features_mod'][keys[1]]
        if bestmodelname == 'LRCV':
            params = bestmodel.get_params()
            bestmodel = LogisticRegressionCV(Cs=params['Cs'], cv=params['cv'], n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        elif bestmodelname == 'SVMCV':
            params = bestmodel.get_params()
            bestmodel = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'], decision_function_shape='ovo', random_state=0).fit(Select_U_data_mod, WHO_train)
        elif bestmodelname == 'KNNCV':
            params = bestmodel.get_params()
            bestmodel = KNeighborsClassifier(n_neighbors=params['n_neighbors'], algorithm=params['algorithm'], leaf_size=params['leaf_size'], metric='minkowski', n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        elif bestmodelname == 'NBCV':
            bestmodel = GaussianNB().fit(Select_U_data_mod, WHO_train)
        elif bestmodelname == 'RF':
            params = bestmodel.get_params()
            bestmodel = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], oob_score=True, n_jobs=-1, random_state=0).fit(Select_U_data_mod, WHO_train)
        elif bestmodelname == 'Stack':
            params = bestmodel.get_params()
            bestmodel = StackingClassifier(estimators=params['estimators'], final_estimator=params['final_estimator'], stack_method='auto', n_jobs=-1).fit(Select_U_data_mod, WHO_train)
        if bestmodelname == 'SVMCV':
            WHO_pred = bestmodel.predict(Select_U_data_test_mod)
        else:
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,1]
        testacc_original = bestmodel.score(Select_U_data_test_mod, WHO_test)
        simplefeaturename = featurename.replace('select_features_','')
        if featurename == 'select_featrues_Caver':
            simplefeaturename = featurename.replace('select_featrues_','')
        fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
        roc_auc = auc(fpr, tpr)
        if (roc_auc < 0.5)&(bestmodelname != 'SVMCV'):
            WHO_pred = bestmodel.predict_proba(Select_U_data_test_mod)[:,0]
            fpr, tpr, thresholds = roc_curve(WHO_test, WHO_pred)
            roc_auc = auc(fpr, tpr)
        if simplefeaturename == 'allmod':
            fprallmod = fpr
            tprallmod = tpr
            roc_aucallmod = roc_auc
            bestmodelnameallmod = bestmodelname
        elif simplefeaturename == 'Aaver':
            fprAaver = fpr
            tprAaver = tpr
            roc_aucAaver = roc_auc
            bestmodelnameAaver = bestmodelname
        plt.plot(fpr, tpr, lw=2, alpha = 0.3, label = 'ROC %s-%s (AUC = %0.2f)' % (simplefeaturename, bestmodelname, roc_auc))
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_tpr = tpr[optimal_idx]
        best_tnr = 1-fpr[optimal_idx]
        best_meanmod_accs.append((simplefeaturename,bestmodelname,round(trainCVacc,4),round(trainCVauc,4), round(testacc_original,4),round(roc_auc,4),round(best_tpr,4),round(best_tnr,4),round(optimal_threshold,4)))
    plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label='Reference Line')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(simplegroupname+' best featuresmodel1-2 test ROC')
    plt.legend(loc = 'lower right', fontsize = 7)
    if simplegroupname == 'allgroups':
        plt.legend(loc="lower right", fontsize=5)
    plt.savefig("./"+simplegroupname+"best_featuresmod1-2testROCs.png")
    plt.show()
    groupsbest_meanmod_accs[simplegroupname] = best_meanmod_accs
    print(simplegroupname + "done.")
joblib.dump(groupsbest_meanmod_accs, "./best_meanmod1-2_accs.pkl")
# 把10个特征一起test的曲线单独画出来
plt.cla()
plt.plot(fprallmod, tprallmod, lw=2, alpha = 0.3, label = 'ROC allmod-%s (AUC = %0.2f)'%(bestmodelnameallmod,roc_aucallmod))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('allmodbest featuresmodel1-2 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./allmodbest_featuresmod1-2_testROCs.png")
plt.show()
#把ADCaver的曲线单独画出来
plt.cla()
plt.plot(fprAaver, tprAaver, lw=2, alpha = 0.3, label = 'ROC Aaver-%s (AUC = %0.2f)'%(bestmodelnameAaver,roc_aucAaver))
plt.plot([0,1], [0,1], color='red', lw=1, linestyle = '--', label = 'Reference Line')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Aaverbest featuresmodel1-2 test ROC')
plt.legend(loc = "lower right", fontsize = 7)
plt.savefig("./Aaverbest_featuresmod1-2_testROCs.png")
plt.show()

print("\n roc 1-2 done.")
#%% 补充一下前面由于第一次疏忽而漏掉的一些数据
#best 2-3 features 1-1和1-2选模过程中的trainCVauc
groups_test2_supple = joblib.load("./groups_rocresults_supple.pkl")

#%%allfeatures 2-1和2-2的Aaver的数据
bestroc_accs = joblib.load("./foldbest2_1accs.pkl")

#%%-------------发现在求estimator平均准确率的时候有的忘记加上s了。。。。。。。。。。。。。。。。。。
select_features_SCs = ALLgroupresults['select_features_SCs']
ALLmodscores = select_features_SCs['ALLmodscores']
Stackscores = []
SVMscores = []
RFscores = []
LRscores = []
KNNscores = []
NBscores = []
for tup in ALLmodscores:
    if 'Stack' in tup[1]:
        Stackscores.append(tup[0])
    elif 'SVM' in tup[1]:
        SVMscores.append(tup[0])
    elif 'RF' in tup[1]:
        RFscores.append(tup[0])
    elif 'LR' in tup[1]:
        LRscores.append(tup[0])
    elif 'KNN' in tup[1]:
        KNNscores.append(tup[0])
    elif 'NB' in tup[1]:
        NBscores.append(tup[0])
Stacksmeanscores = (round(np.mean(Stackscores),4), 'Stackmeanscore')
SVMmeanscores = (round(np.mean(SVMscores),4), 'SVMmeanscore')
RFmeanscores = (round(np.mean(RFscores),4), 'RFmeanscore')
LRmeanscores = (round(np.mean(LRscores),4), 'LRmeanscore')
KNNmeanscores = (round(np.mean(KNNscores),4), 'KNNmeanscore')
NBmeanscores = (round(np.mean(NBscores),4), 'NBmeanscore')
ALLmodmeanscores = [Stacksmeanscores,SVMmeanscores,RFmeanscores,LRmeanscores,KNNmeanscores,NBmeanscores]
ALLmodmeanscores_sorted = sorted(ALLmodmeanscores,reverse=True)

#%%-------------------------------------康康各种设置threshold以后结果有没有一点变化---------------------------------------------------------
foldbest2_1accs_new = joblib.load("./foldbest2_1accs_new.pkl")
#%%
allgroups_new = foldbest2_1accs_new['allgroups']
"""print("External testacc >0.7")
for tup in allgroups_new:
    testacc_final = max(tup[4],tup[5])
    if testacc_final>0.7:
        print(tup)
print("According to inner test results")"""

for tup in allgroups_new:
    if tup[0] in ['F','SI','SC','SCs','SCF','SCFs','allmod']:
        print(tup)
for tup in allgroups_new:
    if 'aver' in tup[0]:
        print(tup)

