# Calculate_features

# Requirements
#   SimpleITK
#   pandas
#   scikit-image
#   numpy
#   pidicom
#   dicom2nifti
#


import SimpleITK as sitk
import pandas as pd
from skimage import draw
import numpy as np
import pydicom
import math
import nibabel
import re
import dicom2nifti
import shutil
import csv
import os
import os.path
from os import path
import glob
import radiomics
from radiomics import featureextractor, imageoperations, firstorder
import six
np.set_printoptions(threshold=np.inf)


class featureCalculator():
    def __init__(self):
        self.csv_file = r'T:\MIP\Katie_Merriman\Project1Data\CalculateTest.csv'
        self.patientFolder = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data'

        ### lambda desktop directory mapping
        #self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/CalculateTest.csv'
        #self.patientFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNormalized_data'

        self.patient_data = []
        self.lesion_data = []


    def calculate(self):
        normalized = []
        errors = []
        #voi_list = []
        PIRADSnames = ['PIRADS', 'pirads', 'PZ', 'TZ']
        PIRADS5names = ['PIRADS_5', 'pirads_5', 'PZ_5', 'TZ_5']
        PIRADS4names = ['PIRADS_4', 'pirads_4', 'PZ_4', 'TZ_4']

        # make list of patients, path to files, and path to individual save folders
        # if save folder doesn't exist for patient, create it
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        patient = []
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['MRN_date']))
            p_path = os.path.join(self.patientFolder, p)
            #save_path = os.path.join(self.saveFolder,p)
            patient.append([p,p_path])
            # If save folder doesn't exist, make it:
            #if os.path.exists(save_path):
            #    continue
            #os.makedirs(save_path)

        # find all voi files and identify wp vs lesions, connect PIRADS score to lesions
        for i in range(0, len(patient)):
            print('Calculating for ', patient[i][0])
            ## look for all VOI files in patient folder
            voi_list = []
            lesion = []
            VOI_levelData = []
            for root, dirs, files in os.walk(patient[i][1]):
                for name in files:
                    PIRADS = 0
                    if name.endswith('wp_bt.nii.gz'):
                        prost_path = os.path.join(root,name)
                    elif name.endswith('bt.nii.gz'):
                        #print('... saving VOI')
                        voiPath = os.path.join(root, name)
                        if any([substring in name for substring in PIRADS5names]):
                            PIRADS = 5
                        elif any([substring in name for substring in PIRADS4names]):
                            PIRADS = 4
                        else:
                            PIRADS = 3
                        voi_list.append([voiPath, PIRADS])
            print('... calculating prostate features')
            prost_info = self.calculateProst(patient_id=patient[i][0], file=prost_path)
            VOI_levelData.append([])
            for j in range(0,len(voi_list)):
                print('... calculating lesion', j,'features')
                lesion_info = self.calculateLesion(patient_id=patient[i][0], file=voi_list[j][0], pirads = voi_list[j][1])
                lesion.append(lesion_info)
                self.lesion_data.append(lesion_info)

            #calculate min/max, total, and relative values
            numLesions = len(lesion)
            p5=[]
            p4=[]
            p3=[]
            index = []
            totalVol = 0
            maxT2ent = 0
            minT2ent = 0
            maxT2uniform=0
            minT2uniform=0
            maxADCent = 0
            minADCent = 0
            maxADCuniform=0
            minADCuniform=0
            maxhighBent = 0
            minhighBent = 0
            maxhighBuniform=0
            minhighBuniform=0
            minADCmean = 0
            minADCmedian = 0
            minADC10 = 0
            minADCIQR = 0
            maxhighBmean = 0
            maxhighBmedian = 0
            maxhighB90=0
            maxhighBIQR = 0
            for k in range(0,len(lesion)):
                if lesion[k][2]==5:
                    p5.append(lesion[k])
                elif lesion[k][2]==4:
                    p4.append(lesion[k])
                else:
                    p3.append(lesion[k])
                if k==0:
                    totalVol = lesion[k][3]
                    maxT2ent = lesion[k][10]
                    minT2ent = lesion[k][10]
                    maxT2uniform = lesion[k][11]
                    minT2uniform = lesion[k][11]
                    maxADCent = lesion[k][16]
                    minADCent = lesion[k][16]
                    maxADCuniform = lesion[k][17]
                    minADCuniform = lesion[k][17]
                    maxhighBent = lesion[k][22]
                    minhighBent = lesion[k][22]
                    maxhighBuniform = lesion[k][23]
                    minhighBuniform = lesion[k][23]
                    minADCmean = lesion[k][12]
                    minADCmedian = lesion[k][13]
                    minADC10 = lesion[k][14]
                    minADCIQR = lesion[k][15]
                    maxhighBmean = lesion[k][18]
                    maxhighBmedian = lesion[k][19]
                    maxhighB90 = lesion[k][20]
                    maxhighBIQR = lesion[k][21]
                else:
                    totalVol = totalVol + lesion[k][3]
                    if lesion[k][10] > maxT2ent:
                        maxT2ent=lesion[k][10]
                    if lesion[k][10] < minT2ent:
                        minT2ent = lesion[k][10]
                    if lesion[k][11]>maxT2uniform:
                        maxT2uniform = lesion[k][11]
                    if lesion[k][11]<minT2uniform:
                        minT2uniform = lesion[k][11]
                    if lesion[k][16]>maxADCent:
                        maxADCent = lesion[k][16]
                    if lesion[k][16]<minADCent:
                        minADCent = lesion[k][16]
                    if lesion[k][17]>maxADCuniform:
                        maxADCuniform = lesion[k][17]
                    if lesion[k][17]<minADCuniform:
                        minADCuniform = lesion[k][17]
                    if lesion[k][22]>maxhighBent:
                        maxhighBent = lesion[k][22]
                    if lesion[k][22]<minhighBent:
                        minhighBent = lesion[k][22]
                    if lesion[k][23]>maxhighBuniform:
                        maxhighBuniform = lesion[k][23]
                    if lesion[k][23]<minhighBuniform:
                        minhighBuniform = lesion[k][23]
                    if lesion[k][12]<minADCmean:
                        minADCmean = lesion[k][12]
                    if lesion[k][13]<minADCmedian:
                        minADCmedian = lesion[k][13]
                    if lesion[k][14]<minADC10:
                        minADC10 = lesion[k][14]
                    if lesion[k][15]<minADCIQR:
                        minADCIQR = lesion[k][15]
                    if lesion[k][18]>maxhighBmean:
                        maxhighBmean = lesion[k][18]
                    if lesion[k][19]>maxhighBmedian:
                        maxhighBmedian = lesion[k][19]
                    if lesion[k][20]>maxhighB90:
                        maxhighB90 = lesion[k][20]
                    if lesion[k][21]>maxhighBIQR:
                        maxhighBIQR = lesion[k][21]

            max=0
            ind=0
            p5Vol = 0
            p4Vol = 0
            p3Vol = 0
            p5VoltoTotal = 0
            p5VoltoProst = 0
            p4VoltoTotal = 0
            p4VoltoProst = 0
            p3VoltoTotal = 0
            p3VoltoProst = 0
            if p5:
                if len(p5)>1:
                    for k in range(0,len(p5)):
                        if p5[k][3]>max:
                            ind = k
                            max = p5[k][3]
                        p5Vol=p5Vol+p5[k][3]
                    index = p5[ind]
                else:
                    p5Vol = p5[0][3]
                    index=p5[0]
                p5VoltoTotal = p5Vol/totalVol
                p5VoltoProst = p5Vol/prost_info[2]

            if p4:
                if len(p4)>1:
                    for k in range(0, len(p4)):
                        if not index:
                            if p4[k][3]>max:
                                ind = k
                                max = p4[k][3]
                        p4Vol=p4Vol+p4[k][3]
                    index = p4[ind]
                else:
                    p4Vol = p4[0][3]
                    if not index:
                        index = p4[0]
                p4VoltoTotal = p4Vol/totalVol
                p4VoltoProst = p4Vol/prost_info[2]
            if p3:
                if len(p3)>1:
                    for k in range(0, len(p3)):
                        if not index:
                            if p3[k][3]>max:
                                ind = k
                                max = p3[k][3]
                        p3Vol=p3Vol+p3[k][3]
                    if not index:
                        index = p3[ind]
                else:
                    p3Vol = p3[0][3]
                    if not index:
                        index = p3[0]
                p3VoltoTotal = p3Vol/totalVol
                p3VoltoProst = p3Vol/prost_info[2]

            patientData = [patient[i][0],numLesions,index[1], prost_info[2],totalVol,p5Vol,p5VoltoTotal,p5VoltoProst,p4Vol,
                           p4VoltoTotal,p4VoltoProst,p3Vol,p3VoltoTotal,p3VoltoProst,index[4],index[5],index[6],index[7],
                           index[8],index[9],prost_info[3],index[10],maxT2ent,minT2ent,prost_info[10],index[16], maxADCent,
                           minADCent, prost_info[17], index[22], maxhighBent, minhighBent, prost_info[4], index[11],
                           maxT2uniform, minT2uniform, prost_info[11], index[17], maxADCuniform, minADCuniform,
                           prost_info[18],index[23],maxhighBuniform,minhighBuniform,prost_info[5],prost_info[6],
                           prost_info[7], prost_info[8], prost_info[9], minADCmean, minADCmedian, minADC10, minADCIQR,
                           index[12], index[13], index[14], index[15], prost_info[12], prost_info[13], prost_info[14],
                           prost_info[15], prost_info[16], maxhighBmean, maxhighBmedian, maxhighB90, maxhighBIQR, index[18],
                           index[19], index[20], index[21]]
            self.patient_data.append(patientData)
            self.lesion_data.append([])




        patientLevelFileName = os.path.join(self.patientFolder, 'MISSINGPatientLevelFeatures.csv')
        patientLevelHeader = ['MRN', 'NumberLesions', 'indexPath', 'prostVol', 'totalVol_all', 'totalVol_P5', 'P5toTotalVol',
                       'P5toProstVol', 'totalVol_P4', 'P4toTotalVol', 'P4toProstVol', 'totalVol_P3', 'P3toTotalVol',
                       'P3toProstVol', 'indexSA', 'indexSphericity', 'indexSVR', 'indexmax3Ddiameter', 'indexElongation',
                       'indexFlatness', 'prostT2_ent', 'indexT2_ent', 'maxT2_ent', 'minT2_ent', 'prostADC_ent',
                       'indexADC_ent', 'maxADC_ent', 'minADC_ent', 'prostHighB_ent', 'indexHighB_ent', 'maxHighB_ent',
                       'minHighB_ent', 'prostT2_uniform', 'indexT2_uniform', 'maxT2_uniform', 'minT2_uniform',
                       'prostADC_uniform', 'indexADC_uniform', 'maxADC_uniform', 'minADC_uniform', 'prostHighB_uniform',
                       'indexHighB_uniform', 'maxHighB_uniform', 'minHighB_uniform', 'prostADC_mean', 'prostADC_median',
                       'prostADC_10th', 'prostADC_90th', 'prostADC_IQR', 'minADC_mean', 'minADC_median', 'minADC_10th',
                       'minADC_IQR', 'indexADC_mean', 'indexADC_median', 'indexADC_10th', 'indexADC_IQR',
                       'prostHighB_mean', 'prostHighB_median', 'prostHighB_10th', 'prostHighB_90th', 'prostHighB_IQR',
                       'maxHighB_mean', 'maxHighB_median', 'maxHighB_90th', 'maxHighB_IQR', 'indexHighB_mean',
                       'indexHighB_median', 'indexHighB_90th', 'indexHighB_IQR']
        with open(patientLevelFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(patientLevelHeader)
            csvwriter.writerows(self.patient_data)


        LesionLevelFileName = os.path.join(self.patientFolder, 'MISSINGLesionLevelFeatures.csv')
        LesionHeader = ['MRN', 'FilePath', 'PIRADS', 'Volume','SurfaceArea', 'Sphericity', 'SVR', '3Ddiameter',
                        'Elongation', 'Flatness', 'T2entropy', 'T2uniformity', 'ADCmean','ADCmedian','ADC10', 'ADCIQR',
                        'ADCent', 'ADC_uniformity', 'HighBmean', 'HighBmedian', 'HighB90', 'HighB_IQR','HighB_entropy',
                        'HighB_uniformity']
        with open(LesionLevelFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(LesionHeader)
            csvwriter.writerows(self.lesion_data)








    def calculateProst(self, patient_id='',file=''):
        rad_info = [patient_id, file]

        # load t2 and adc niftis, create arrays
        t2_img_path = os.path.join(self.patientFolder, patient_id, 'T2n.nii')
        t2_img = sitk.ReadImage(t2_img_path)
        #img_array = sitk.GetArrayFromImage(t2_img)
        #img_array = np.swapaxes(img_array, 2, 0)

        adc_img_path = os.path.join(self.patientFolder, patient_id, 'ADCn.nii')
        adc_img = sitk.ReadImage(adc_img_path)
        #adc_array = sitk.GetArrayFromImage(adc_img)
        #adc_array = np.swapaxes(adc_array, 2, 0)

        highB_img_path = os.path.join(self.patientFolder, patient_id, 'highBn.nii')
        highB_img = sitk.ReadImage(highB_img_path)
        #highB_array = sitk.GetArrayFromImage(adc_img)
        #highB_array = np.swapaxes(highB_array, 2, 0)

        mask_img = sitk.ReadImage(file)

        applyLog = False
        applyWavelet = False

        # Setting for the feature calculation.
        # Currently, resampling is disabled.
        # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
        settings = {'binWidth': 25,
                    'interpolator': sitk.sitkBSpline,
                    'resampledPixelSpacing': None}

        interpolator = settings.get('interpolator')
        resampledPixelSpacing = settings.get('resampledPixelSpacing')
        if interpolator is not None and resampledPixelSpacing is not None:
            adc_img, mask_img = imageoperations.resampleImage(t2_img, mask_img, **settings)


        ### Get T2 values
        # mask_SA = radiomics.shape.RadiomicsShape(t2_img, img_out).getSurfaceAreaFeatureValue()
        mask_vol = 0.001 * radiomics.shape.RadiomicsShape(t2_img, mask_img).getMeshVolumeFeatureValue()
        # mask_SVR = radiomics.shape.RadiomicsShape(t2_img, img_out).getSurfaceVolumeRatioFeatureValue()
        # mask_sphr = radiomics.shape.RadiomicsShape(t2_img, img_out).getSphericityFeatureValue()
        # mask_3Dd = radiomics.shape.RadiomicsShape(t2_img, img_out).getMaximum3DDiameterFeatureValue()
        # mask_elong = radiomics.shape.RadiomicsShape(t2_img, img_out).getElongationFeatureValue()
        # mask_flat = radiomics.shape.RadiomicsShape(t2_img, img_out).getFlatnessFeatureValue()
        rad_info.append(mask_vol)
        #rad_info.extend([mask_SA, mask_vol, mask_SVR, mask_sphr, mask_3Dd, mask_elong, mask_flat])

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(t2_img, mask_img, **settings)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)


        results = firstOrderFeatures.execute()
        #rad_info.append(float(results["Entropy"]))
        for (key, val) in six.iteritems(results):
            rad_info.append(float(val))

        ## collect radiomics shape info

        ### Get ADC mean, median, 10Percentile, and Entropy values
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(adc_img, mask_img, **settings)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('10Percentile', True)
        firstOrderFeatures.enableFeatureByName('90Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)
        # firstOrderFeatures.enableAllFeatures()

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            # print('  ', key, ':', val)
            rad_info.append(float(val))

        ### Get highB mean, median, 10th Percentile, 90thPercentile, Uniformity and Entropy values
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(highB_img, mask_img, **settings)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('10Percentile', True)
        firstOrderFeatures.enableFeatureByName('90Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            # print('  ', key, ':', val)
            rad_info.append(float(val))

        return rad_info

    def calculateLesion(self, patient_id='',file='', pirads=''):
        rad_info = [patient_id, file, pirads]

        # load t2 and adc niftis, create arrays
        t2_img_path = os.path.join(self.patientFolder, patient_id, 'T2n.nii')
        t2_img = sitk.ReadImage(t2_img_path)
        #img_array = sitk.GetArrayFromImage(t2_img)
        #img_array = np.swapaxes(img_array, 2, 0)

        adc_img_path = os.path.join(self.patientFolder, patient_id, 'ADCn.nii')
        adc_img = sitk.ReadImage(adc_img_path)
        #adc_array = sitk.GetArrayFromImage(adc_img)
        #adc_array = np.swapaxes(adc_array, 2, 0)

        highB_img_path = os.path.join(self.patientFolder, patient_id, 'highBn.nii')
        highB_img = sitk.ReadImage(highB_img_path)
        #highB_array = sitk.GetArrayFromImage(adc_img)
        #highB_array = np.swapaxes(highB_array, 2, 0)

        mask_img = sitk.ReadImage(file)

        applyLog = False
        applyWavelet = False

        # Setting for the feature calculation.
        # Currently, resampling is disabled.
        # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
        #settings = {'binWidth': 25,
        #            'interpolator': sitk.sitkBSpline,
        #            'resampledPixelSpacing': None}

        #interpolator = settings.get('interpolator')
        #resampledPixelSpacing = settings.get('resampledPixelSpacing')
        #if interpolator is not None and resampledPixelSpacing is not None:
        #    t2_img, mask_img = imageoperations.resampleImage(t2_img, mask_img, **settings)


        ### Get T2 values
        mask_vol = 0.001 * radiomics.shape.RadiomicsShape(t2_img, mask_img).getMeshVolumeFeatureValue()
        mask_SA = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSurfaceAreaFeatureValue()
        mask_sphr = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSphericityFeatureValue()
        mask_SVR = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSurfaceVolumeRatioFeatureValue()
        mask_3Dd = radiomics.shape.RadiomicsShape(t2_img, mask_img).getMaximum3DDiameterFeatureValue()
        mask_elong = radiomics.shape.RadiomicsShape(t2_img, mask_img).getElongationFeatureValue()
        mask_flat = radiomics.shape.RadiomicsShape(t2_img, mask_img).getFlatnessFeatureValue()
        #T2ent = radiomics.firstorder.RadiomicsFirstOrder(t2_img, mask_img).getEntropyFeatureValue()
        #T2unif = radiomics.firstorder.RadiomicsFirstOrder(t2_img, mask_img).getUniformityFeatureValue()
        #rad_info.append(mask_vol)
        rad_info.extend([mask_vol, mask_SA, mask_sphr, mask_SVR, mask_3Dd, mask_elong, mask_flat])
        #rad_info.extend([T2ent, T2unif])

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(t2_img, mask_img)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)


        results = firstOrderFeatures.execute()
        #rad_info.append(float(results["Entropy"]))
        for (key, val) in six.iteritems(results):
            rad_info.append(float(val))



        #if interpolator is not None and resampledPixelSpacing is not None:
        #    adc_img, mask_img = imageoperations.resampleImage(adc_img, mask_img, **settings)

        ## collect radiomics shape info
        # ADCmean = radiomics.firstorder.RadiomicsFirstOrder(t2_img, img_out).getMeanFeatureValue()


        ### Get ADC mean, median, 10Percentile, and Entropy values
        #ADCmean = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).getMeanFeatureValue()
        #ADCmed = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).getMedianFeatureValue()
        #ADC10 = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).get10PercentileFeatureValue()
        #ADCIQR = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).getInterquartileRangeFeatureValue()
        #ADCent = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).getEntropyFeatureValue()
        #ADCunif = radiomics.firstorder.RadiomicsFirstOrder(adc_img, mask_img).getUniformityFeatureValue()
        #rad_info.extend([ADCmean, ADCmed, ADC10, ADCIQR, ADCent, ADCunif])

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(adc_img, mask_img)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('10Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)
        #firstOrderFeatures.enableAllFeatures()

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
            rad_info.append(float(val))


        #if interpolator is not None and resampledPixelSpacing is not None:
        #    highB_img, mask_img = imageoperations.resampleImage(highB_img, mask_img, **settings)

        ### Get highB mean, median, 10th Percentile, 90thPercentile, Uniformity and Entropy values
        #highBmean = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).getMeanFeatureValue()
        #highBmed = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).getMedianFeatureValue()
        #highB10 = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).get10PercentileFeatureValue()
        #highBIQR = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).getInterquartileRangeFeatureValue()
        #highBent = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).getEntropyFeatureValue()
        #highBunif = radiomics.firstorder.RadiomicsFirstOrder(highB_img, mask_img).getUniformityFeatureValue()
        #rad_info.extend([highBmean, highBmed, highB10, highBIQR, highBent, highBunif])


        firstOrderFeatures = firstorder.RadiomicsFirstOrder(highB_img, mask_img)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('90Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
            rad_info.append(float(val))

        return rad_info


if __name__ == '__main__':
    c = featureCalculator()
    c.calculate()
    print('Conversions complete')


















