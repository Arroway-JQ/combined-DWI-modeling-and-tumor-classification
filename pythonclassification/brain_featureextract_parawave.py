#%%
# import the python libs
import SimpleITK as sitk
from radiomics import featureextractor,imageoperations,firstorder
import numpy as np
import os
import sys
from scipy import stats
import pandas as pd
import joblib
print("preparing labs done.\n")

#%%  Initializing the  extractor
paramPath = os.path.join('../pyradiomics-master/','examples','exampleSettings','Params.yaml')
print('Parameter file, absolute path:',os.path.abspath(paramPath))
# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
print('Extraction parameters:\n\t',extractor.settings)
print('Enabled filters:\n\t',extractor.enabledImagetypes)
print('Enabled features:\n\t',extractor.enabledFeatures)
#%% load the beda RENMIN data for 3 DWI series
mainDir = "E:/brain/Usingdata_Huashan/adultsdata/"
objects = {}
DWIseries = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
for patient in os.listdir(mainDir):
    patientDir = os.path.join(mainDir,patient)
    ImgDir = patientDir + '/radiomicsdata/'
    features = {}
    wavelet = {}
    print("Extracting %s:\n"%(patient))
    for DWI in DWIseries:
        MainimgDir = ImgDir+'Eddycor_brain'+str(DWI)+'.nii'
        bMainimgnii = sitk.ReadImage(MainimgDir)
        labelnii = sitk.ReadImage(ImgDir+'label.nii')
        try:
            features[DWI] = extractor.execute(bMainimgnii, labelnii)
        except Exception as e:
            labelimg = sitk.GetArrayFromImage(labelnii)

            bMainimg = sitk.GetArrayFromImage(bMainimgnii)
            s_m = np.shape(bMainimg)
            s_l = np.shape(labelimg)
            labelimgnew = np.zeros((s_m[0], s_m[1], s_m[2]), int)
            for i in range(s_l[0]):
                labelimgnew[i] = labelimg[i]
            labelout = sitk.GetImageFromArray(labelimgnew)
            labelout.SetSpacing(bMainimgnii.GetSpacing())
            labelout.SetOrigin(bMainimgnii.GetOrigin())
            sitk.WriteImage(labelout,ImgDir + 'label.nii')
            print("label error:"+patient)
            labelnii = sitk.ReadImage(ImgDir+'label.nii')
            features[DWI] = extractor.execute(bMainimgnii,labelnii)
        print(str(DWI)+' calculate params features done.\n')
        #对每一幅图片加入小波变换的特征
        waveletFeatures = {}
        bb, correctedMask = imageoperations.checkMask(bMainimgnii,labelnii, LABEL=1)
        for decompositionImage, decompositionName, inputSettings in imageoperations.getWaveletImage(bMainimgnii,labelnii):
            decompositionImage,croppedMask = imageoperations.cropToTumorMask(decompositionImage,labelnii,bb)
            waveletFirstOrderFeatures = firstorder.RadiomicsFirstOrder(decompositionImage,croppedMask, **inputSettings)
            waveletFirstOrderFeatures.enableAllFeatures()
            print('Calculate firstorder features with ',decompositionName)
            waveletFeatures[decompositionName] = waveletFirstOrderFeatures.execute()
        wavelet[DWI] = waveletFeatures
    objects[patient] = (features,wavelet)
    # 至此，objects应该包含92个病人，每个病人共有features=100, waveletFeatures =8*18=144, 理论上一共提取了3X(100+144)=732个特征
OtherDir = './featuresselection/'
joblib.dump(objects,OtherDir+"brain_paramwave_features.pkl")
features_names = list(sorted(filter(lambda k: k.startswith("original_"),features[0])))

#%% 定义一下energy和entropy
def Entropy(labels, base=2):
    probs = pd.Series(labels).value_counts()/len(labels)
    en =  stats.entropy(probs,base=base)
    return en
#%%
feature_series_name = ['max','min','mean','var','median','kurtosis','skewness','energy']

def comput_feature_series(DWIseries,features_names,wavelet):
    samples = np.zeros((22,244))
    i = 0
    for DWI in DWIseries:
        a = np.array([])
        for feature_name in features_names:
            a = np.append(a,features[DWI][feature_name])
        for wavelet_name in wavelet[DWI].keys():
            for waveletfirst_name in wavelet[DWI][wavelet_name].keys():
                a = np.append(a,wavelet[DWI][wavelet_name][waveletfirst_name])
        samples[i,:] = a
        i = i+1
    MAX = samples.max(axis=0)
    MIN = samples.min(axis=0)
    MEAN = samples.mean(axis=0)
    VAR = np.var(samples,axis=0)
    MEDIAN = np.median(samples,axis =0)
    KURTOSIS = stats.kurtosis(samples,axis=0)
    SKEWNESS = stats.skew(samples,axis=0)
    ENERGY = np.sum(np.power(samples,2),axis=0)
    resamples = samples.flatten()
    newsamples = np.hstack((np.hstack((np.hstack((np.hstack((np.hstack((np.hstack((np.hstack((np.hstack((resamples,MAX)),MIN)),MEAN)),VAR)),MEDIAN)),KURTOSIS)),SKEWNESS)),ENERGY))
    return newsamples
#%% 循环所有的case
featuresmap = np.zeros((74,7320))  #(22+8)*(100+144)= 7320
p = 0
for patient in objects.keys():
    print("Dealing with %s...\n"%(patient))
    features = objects[patient][0]
    wavelet=objects[patient][1]
    newsamples = comput_feature_series(DWIseries,features_names,wavelet)
    featuresmap[p,:] = newsamples
    p = p+1
#我觉得我最好把featurename做出来保存一下，特征太多了，怕到时候分不清楚
features_namescopy = features_names.copy()
featurenames = []
wavelet = objects[patient][1][0]
for DWI in DWIseries:
    for name in features_namescopy:
        firstfeaturename = str(DWI) + "_" +name
        featurenames.append(firstfeaturename)

    for key in wavelet:
        subfeature = wavelet[key]
        simplekey = key.replace('wavelet-','')
        for k in subfeature:
            wavefeature_name = str(DWI)+"_"+simplekey+'_'+k
            featurenames.append(wavefeature_name)

for key in wavelet:
    subfeature = wavelet[key]
    simplekey = key.replace('wavelet-','')
    for k in subfeature:
        wavefeaturename = simplekey+'_'+k
        features_namescopy.append(wavefeaturename)
seriesnames = []
for seriesname in feature_series_name:
    for name in features_namescopy:
        sname= seriesname+"_"+name
        seriesnames.append(sname)
featurenames.extend(seriesnames)
np.shape(featurenames)

features_comp = {'featurenames':featurenames,'featuresmap':featuresmap}
joblib.dump(features_comp,OtherDir+'ori_radiomicfeaturesmap_brain.pkl')

#%%测试一下entropy
samples = np.zeros((22,244))
wavelet = objects[patient][1]
i = 0
for DWI in DWIseries:
    a = np.array([])
    for feature_name in features_names:
        a = np.append(a,features[DWI][feature_name])
    for wavelet_name in wavelet[DWI].keys():
        for waveletfirst_name in wavelet[DWI][wavelet_name].keys():
            a = np.append(a,wavelet[DWI][wavelet_name][waveletfirst_name])
    samples[i,:] = a
    i = i+1
data1  = samples[:,5]
entropyn1 = Entropy(data1,2)
data2 = samples[:,10]
entropyn2 = Entropy(data2,2)
data1_ = samples[:,5]
entropyn1_ = Entropy(data1_)
data2_ = samples[:,10]
entropyn2_ = Entropy(data2_)
print("entropy1: %f"%(entropyn1))
print("entropy2: %f"%(entropyn2))
print("entropy1_: %f"%(entropyn1_))
print("entropy2_: %f"%(entropyn2_))
#我觉着这个entropy就是个摆设，那么多值肯定都不完全相等啊，这样每个人的entropy都是一样的还区分个毛线啊，区分了个寂寞，删掉删掉。。。。。。。。。。。。



