#%% load liabrary
import sys
sys.path.append("/")
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import Lasso, Ridge, LassoCV,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
print("Import library done.\n")
#%%
#%%循环37个特征组合，其中后7个组合为经典模型组合
select_features = joblib.load("./select_features.pkl")
datapreparing = joblib.load('./datapreparing.pkl')
traindataall = datapreparing['traindataall']
testdataall = datapreparing['testdataall']
WHO_train = traindataall['WHO']
WHO_test = testdataall['WHO']
print("Preparing data done.")

ALLgroup_bestmodscore = []
ALLgroup_bestmodmeanscore = []
ALLgroupresults = {}
for key in select_features.keys():
    val = select_features[key]
    select_features_mod = val
    select_U_features_mod = list(val.values())[0]
    select_U_data_mod = list(val.values())[1]
    select_U_data_test_mod = list(val.values())[2]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

    k = 0
    ALLmodscores = []
    ALLmod = {}
    LRscores = []
    SVMscores = []
    KNNscores = []
    NBscores = []
    RFscores = []
    Stackscores = []
    Lassoscores = []

    param_grid_svm = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10, 50], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    gsearch_svm = GridSearchCV(svm.SVC(), param_grid_svm, scoring='accuracy', cv=5, n_jobs=-1).fit(select_U_data_mod, WHO_train)
    best_param_svm = gsearch_svm.best_params_
    ker = best_param_svm['kernel']
    bestC = best_param_svm['C']
    bestgamma = best_param_svm['gamma']
    print("Parameter search for SVM done.")

    param_grid_knn = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': [30, 35, 40, 45, 50]}
    gsearch_knn = GridSearchCV(estimator=KNeighborsClassifier(p=2, metric='minkowski'), param_grid=param_grid_knn, cv=5,scoring='accuracy', n_jobs=-1).fit(select_U_data_mod, WHO_train)
    best_param_knn = gsearch_knn.best_params_
    bestn = best_param_knn['n_neighbors']
    bestalgo = best_param_knn['algorithm']
    bestleaf = best_param_knn['leaf_size']
    print("Parameter search for KNN done.")

    param_grid_RF = {'n_estimators': [10, 100, 1000, 2000, 10000], 'max_depth': [2, 3, 5, 10, 15, 20, 25, 30],
                      'min_samples_split': [2, 4, 6, 8, 10, 20, 30],
                      'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]}
    gsearch_RF = GridSearchCV(estimator=RandomForestClassifier(oob_score=True, random_state=0), param_grid=param_grid_RF, cv=5, scoring='accuracy', n_jobs=-1).fit(select_U_data_mod, WHO_train)
    bestn2 = gsearch_RF.best_params_['n_estimators']
    bestdepth2 = gsearch_RF.best_params_['max_depth']
    bestminsplit = gsearch_RF.best_params_['min_samples_split']
    bestminleaf = gsearch_RF.best_params_['min_samples_leaf']
    print("Parameter search for RF done.")

    for train_index, test_index in rskf.split(select_U_data_mod, WHO_train):
        X_train, X_test = select_U_data_mod[train_index,:], select_U_data_mod[test_index,:]
        Y_train, Y_test = WHO_train[train_index], WHO_train[test_index]
        k = k + 1
        #LR model
        LRmodelname = 'LRfold'+str(k)
        LRCV = LogisticRegressionCV(cv=5, random_state=0, n_jobs=-1).fit(X_train,Y_train)
        LRscore = LRCV.score(X_test,Y_test)
        LRscores.append(LRscore)
        ALLmodscores.append((LRscore, LRmodelname))
        LRmodelresutls = {'LRCV':LRCV, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[LRmodelname] = LRmodelresutls
        print(LRmodelname + 'done.')
        #SVM model
        SVMmodelname = 'SVMfold'+str(k)
        SVMCV = svm.SVC(C=bestC, kernel=ker, gamma=bestgamma,decision_function_shape='ovo', random_state=0).fit(X_train, Y_train)
        SVMscore = SVMCV.score(X_test,Y_test)
        SVMscores.append(SVMscore)
        ALLmodscores.append((SVMscore, SVMmodelname))
        SVMmodelresutls = {'SVMCV':SVMCV, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[SVMmodelname] = SVMmodelresutls
        print(SVMmodelname + 'done.')
        #KNN model
        KNNmodelname = 'KNNfold'+str(k)
        KNNCV = KNeighborsClassifier(n_neighbors=bestn, algorithm=bestalgo, leaf_size=bestleaf, p=2, metric='minkowski').fit(X_train,Y_train)
        KNNscore = KNNCV.score(X_test, Y_test)
        KNNscores.append(KNNscore)
        ALLmodscores.append((KNNscore, KNNmodelname))
        KNNmodelresults = {'KNNCV':KNNCV, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[KNNmodelname] = KNNmodelresults
        print(KNNmodelname + 'done.')
        #NB model
        NBmodelname = 'NBfold'+str(k)
        NBCV = GaussianNB().fit(X_train,Y_train)
        NBscore = NBCV.score(X_test,Y_test)
        NBscores.append(NBscore)
        ALLmodscores.append((NBscore, NBmodelname))
        NBmodelresults = {'NBCV':NBCV, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[NBmodelname] = NBmodelresults
        print(NBmodelname + 'done.')
        #RF model
        RFmodelname = 'RFfold' + str(k)
        RFCV = RandomForestClassifier(n_estimators=bestn2, max_depth=bestdepth2, min_samples_split=bestminsplit, min_samples_leaf=bestminleaf, oob_score=True, random_state=0, n_jobs=-1).fit(X_train, Y_train)
        RFscore = RFCV.score(X_test,Y_test)
        RFscores.append(RFscore)
        ALLmodscores.append((RFscore, RFmodelname))
        RFmodelresults = {'RF':RFCV, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[RFmodelname] = RFmodelresults
        print(RFmodelname + 'done.')
        #Stacking model
        Stackmodelname = 'Stackfold' + str(k)
        clf1 = GaussianNB()
        clf2 = RandomForestClassifier(n_estimators=bestn2, max_depth=bestdepth2, min_samples_split=bestminsplit, min_samples_leaf=bestminleaf, oob_score=True, random_state=0, n_jobs=-1)
        clf3 = KNeighborsClassifier(n_neighbors=bestn, algorithm=bestalgo, leaf_size=bestleaf, p=2, metric='minkowski')
        clf4 = svm.SVC(C=bestC, kernel=ker, gamma=bestgamma,decision_function_shape='ovo', random_state=0)
        estimator = [('NB',clf1),('RF',clf2),('KNN',clf3),('SVM',clf4)]
        clf5 = StackingClassifier(estimators=estimator, final_estimator=LogisticRegressionCV(cv=5,random_state=0), stack_method='auto', n_jobs=-1).fit(X_train,Y_train)
        Stackscore = clf5.score(X_test,Y_test)
        Stackscores.append(Stackscore)
        ALLmodscores.append((Stackscore,Stackmodelname))
        Stackmodelresutls = {'Stack':clf5, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}
        ALLmod[Stackmodelname] = Stackmodelresutls
        print(Stackmodelname + 'done.')
    ALLmodscores_sorted = sorted(ALLmodscores, reverse = True)
    LRmeanscore = (round(np.mean(LRscores),4), 'LRmeanscore')
    SVMmeansore = (round(np.mean(SVMscores),4), 'SVMmeanscore')
    KNNmeanscore = (round(np.mean(KNNscores),4), 'KNNmeanscore')
    NBmeanscore = (round(np.mean(NBscores),4), 'NBmeanscore')
    RFmeanscore = (round(np.mean(RFscores),4), 'RFmeanscore')
    Stackmeanscore = (round(np.mean(Stackscores),4), 'Stackmeanscore')
    ALLmodmeanscores = [LRmeanscore, SVMmeansore, KNNmeanscore, NBmeanscore, RFmeanscore, Stackmeanscore]
    ALLmodmeanscores_sorted = sorted(ALLmodmeanscores, reverse = True)
    groupresults = {'ALLmod':ALLmod, 'ALLmodscores':ALLmodscores_sorted, 'ALLmodmeanscores':ALLmodmeanscores_sorted}
    bestmodscore = ALLmodscores_sorted[0][0]
    bestmodmeanscore = ALLmodmeanscores_sorted[0][0]
    ALLgroup_bestmodscore.append((bestmodscore,key))
    ALLgroup_bestmodmeanscore.append((bestmodmeanscore,key))
    ALLgroupresults[key] = groupresults
    print('\n' + key + 'done')
ALLgroup_bestmodscore_sorted = sorted(ALLgroup_bestmodscore, reverse=True)
ALLgroup_bestmodmeanscore_sorted = sorted(ALLgroup_bestmodmeanscore, reverse=True)
ALLgroups = {'ALLgroup_bestmodscore':ALLgroup_bestmodscore_sorted, 'ALLgroup_bestmodmeanscore':ALLgroup_bestmodmeanscore_sorted, 'ALLgroupresults':ALLgroupresults}
joblib.dump(ALLgroups, "./ALLgroups.pkl")