## Radiomics_Systematic_Lesions.py
## Katie Merriman
## Written 2/27/23
## Takes in csv with patient MRNs and filepath of folder containing all NIfTIs and masks
## Can switch between filepaths for running locally or remotely by changing variable "local" at beginning of class definition
## Divides prostate into 12 sections following transrectal systematic biopsy regions
## Records number and name of manually contoured lesions present in each section
## Records highest PI-RADS score present in each section (0 = none, 6 = whole prostate lesion)
## Collects ALL available PyRadiomics features for:
## 1. Manually contoured whole prostate
## 2. Manually contoured lesions
## 3. Systematically divided prostate sections
## Creates then appends .csv file with radiomics data after every patient



import pandas as pd
import os
import os.path
from os import path
import csv
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
import glob
import radiomics
#from radiomics import featureextractor, imageoperations, firstorder, glcm, glszm, ngtdm
import six
np.set_printoptions(threshold=np.inf)


class featureCalculator():
    def __init__(self):
        local = 1

        if local:
            self.csv_file = r'T:\MIP\Katie_Merriman\RadiomicsProject\Patients4Radiomics_test.csv'
            self.patientFolder = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data'
            self.saveFolder = r'T:\MIP\Katie_Merriman\RadiomicsProject'

        else:
        ### lambda desktop directory mapping
            self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/RadiomicsProject/Patients4Radiomics_test.csv'
            self.patientFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNormalized_data'
            self.saveFolder = 'Mdrive_mount/MIP/Katie_Merriman/RadiomicsProject'


        #self.patient_data = []
        #self.lesion_data = []

        self.PIRADSnames = ['PIRADS', 'pirads', 'PZ', 'TZ']
        self.PIRADS5names = ['PIRADS_5', 'pirads_5', 'PZ_5', 'TZ_5']
        self.PIRADS4names = ['PIRADS_4', 'pirads_4', 'PZ_4', 'TZ_4']
        self.PIRADS3names = ['PIRADS_3', 'pirads_3', 'PZ_3', 'TZ_3']
        self.wpLesion = '1_PIRADS_1_bt.nii'


    def calculate(self):
        #errors = []
        #voi_list = []

        """
        ## INITIALIZE NEW CSV FILE ##
        # Note: Single row of info needs to be in double brackets or each character will get its own cell
        headers = [['MRN', '']]
        # open csv file in 'a+' mode to append
        file = open(r'T:\MIP\Katie_Merriman\RadiomicsProject\Systematic_and_Lesion_Radiomics.csv', 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows(headers)
        """

        # make list of patients, path to files
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        patient = []
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['MRN_date']))
            p_path = os.path.join(self.patientFolder, p)
            patient.append([p, p_path])

        ### FOR EACH PATIENT: ###
        for i in range(0, len(patient)):
            prost_data = []
            VOI_data = []
            segment_data = []

            ### CALCULATE RADIOMICS FEATURES ###
            try:
            ## calculate whole prostate features
                prost_data = self.calculateProst(patient[i])

                ## identify VOIs with PIRADS
                voi_list = self.getVOIlist(patient[i])

                ## calculate segment features
                segment_data = self.calculateSegment(patient[i], voi_list, prost_data)

                ## calculate lesion features
                VOI_data = self.calculateLesion(patient[i], voi_list, prost_data[1])

                ### UPDATE CSV WITH PATIENT RADIOMICS INFO ###
                radiomics_data = [[patient[i][0]+prost_data+segment_data+VOI_data]]
                # open csv file in 'a+' mode to append

                ##########IF THE FILE DOESN'T EXIST, COMBINE 1st LINE OF EACH 


                file = open(r'T:\MIP\Katie_Merriman\RadiomicsProject\Systematic_and_Lesion_Radiomics.csv', 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows(radiomics_data)
            except FileNotFoundError:
                file = open(r'T:\MIP\Katie_Merriman\RadiomicsProject\Systematic_and_Lesion_Radiomics.csv', 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient[i][0],'Error: File not found']])



    def calculateProst(self,patient):
        prost_data = []
        wpMask = os.path.join(patient[1], patient[0], 'wp_bt.nii.gz')  # path join patient path + mrn_with_date + wp_bt.nii.gz
        mask_img = sitk.ReadImage(wpMask)

        prost_data = self.calculateRadiomics(patient, "wp", mask_img)
        return prost_data

    def getVOIlist(self, patient):
        voi_list = []
        for root, dirs, files in os.walk(patient[1]):
            for name in files:
                PIRADS = 0
                if name.endswith('bt.nii.gz'):
                    if not name.endswith('wp_bt.nii.gz'):
                        voiPath = os.path.join(root, name)
                        if any([substring in name for substring in PIRADS5names]):
                            PIRADS = 5
                        elif any([substring in name for substring in PIRADS4names]):
                            PIRADS = 4
                        else:
                            PIRADS = 3
                        voi_list.append([voiPath, name, PIRADS])
        return voi_list

    def calculateSegment(self, patient, voi_list, prost_data):
        seg_data = []
        wpMask = os.path.join(patient[1], patient[0], 'wp_bt.nii.gz') # path join patient path + mrn_with_date + wp_bt.nii.gz
        mask_img = sitk.ReadImage(wpMask)


        #mask_img = sitk.ReadImage(r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\wp_bt.nii.gz')
        mask_arr = sitk.GetArrayFromImage(mask_img)
        mskNZ = mask_arr.nonzero()

        ### CREATE BASE/MID/APEX MASKS ###
        ## Find axial boundaries  of each section by dividing slices containing prostate by 3
        lowerZ = int(min(mskNZ[0]) + round((max(mskNZ[0]) - min(mskNZ[0])) / 3))
        UpperZ = int(max(mskNZ[0]) - round(max(mskNZ[0]) - min(mskNZ[0])) / 3)

        ## Create blank masks and arrays for base (upper), mid, and apex (lower)
        LowerMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        LowerArr = sitk.GetArrayFromImage(LowerMask)
        MidMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        MidArr = sitk.GetArrayFromImage(MidMask)
        UpperMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        UpperArr = sitk.GetArrayFromImage(UpperMask)

        # populate each blank array with 1s where original mask is 1 within each axial section boundary
        for index in range(len(mskNZ[0])):
            if mskNZ[0][index] < lowerZ:
                LowerArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1
            elif mskNZ[0][index] < UpperZ:
                MidArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1
            else:
                UpperArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1



        #for nz in range(mskNZ[0].tolist().index(lowerZ)):
        #    LowerArr[mskNZ[0][nz], mskNZ[1][nz], mskNZ[2][nz]] = 1


        ## Write masks to file to check function
        LowerMask = sitk.GetImageFromArray(LowerArr)
        LowerMask.CopyInformation(mask_img)
        sitk.WriteImage(LowerMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\lower.nii.gz')

        MidMask = sitk.GetImageFromArray(MidArr)
        MidMask.CopyInformation(mask_img)
        sitk.WriteImage(MidMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\mid.nii.gz')

        UpperMask = sitk.GetImageFromArray(UpperArr)
        UpperMask.CopyInformation(mask_img)
        sitk.WriteImage(UpperMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\upper.nii.gz')






        ## Divide each section into anterior/posterior and left/right quadrants

        ## Apex
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        LowerArrNZ = LowerArr.nonzero()
        LowCenterX = int(min(LowerArrNZ[1] + round((max(LowerArrNZ[1]) - min(LowerArrNZ[1])) / 2)))
        LowCenterY = int(min(LowerArrNZ[2] + round((max(LowerArrNZ[2]) - min(LowerArrNZ[2])) / 2)))

        # Create blank masks/arrays
        LowAntRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        LowAntRArr = sitk.GetArrayFromImage(LowAntRMask)
        LowAntLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        LowAntLArr = sitk.GetArrayFromImage(LowAntLMask)
        LowPostRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        LowPostRArr = sitk.GetArrayFromImage(LowPostRMask)
        LowPostLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        LowPostLArr = sitk.GetArrayFromImage(LowPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(LowerArrNZ[2])):
            if LowerArrNZ[1][index] < LowCenterX: # if anterior
                if LowerArrNZ[2][index] < LowCenterY: # if right anterior
                    LowAntRArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
                else: #else left anterior
                    LowAntLArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
            elif LowerArrNZ[2][index] < LowCenterY: # else posterior, if right posterior
                LowPostRArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
            else: # else left posterior
                LowPostLArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        LowAntRMask = sitk.GetImageFromArray(LowAntRArr)
        LowAntRMask.CopyInformation(mask_img)
        #sitk.WriteImage(LowAntRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\LowAntR.nii.gz')

        LowAntLMask = sitk.GetImageFromArray(LowAntLArr)
        LowAntLMask.CopyInformation(mask_img)
        #sitk.WriteImage(LowAntLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\LowAntL.nii.gz')

        LowPostRMask = sitk.GetImageFromArray(LowPostRArr)
        LowPostRMask.CopyInformation(mask_img)
        #sitk.WriteImage(LowPostRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\LowPostR.nii.gz')

        LowPostLMask = sitk.GetImageFromArray(LowPostLArr)
        LowPostLMask.CopyInformation(mask_img)
        #sitk.WriteImage(LowPostLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\LowPostL.nii.gz')

        ## Mid
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        MidArrNZ = MidArr.nonzero()
        MidCenterX = int(min(MidArrNZ[1] + round((max(MidArrNZ[1]) - min(MidArrNZ[1])) / 2)))
        MidCenterY = int(min(MidArrNZ[2] + round((max(MidArrNZ[2]) - min(MidArrNZ[2])) / 2)))

        # Create blank masks/arrays
        MidAntRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        MidAntRArr = sitk.GetArrayFromImage(MidAntRMask)
        MidAntLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        MidAntLArr = sitk.GetArrayFromImage(MidAntLMask)
        MidPostRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        MidPostRArr = sitk.GetArrayFromImage(MidPostRMask)
        MidPostLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        MidPostLArr = sitk.GetArrayFromImage(MidPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(MidArrNZ[2])):
            if MidArrNZ[1][index] < MidCenterX:  # if anterior
                if MidArrNZ[2][index] < MidCenterY:  # if right anterior
                    MidAntRArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
                else:  # else left anterior
                    MidAntLArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
            elif MidArrNZ[2][index] < MidCenterY:  # else posterior, if right posterior
                MidPostRArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
            else:  # else left posterior
                MidPostLArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        MidAntRMask = sitk.GetImageFromArray(MidAntRArr)
        MidAntRMask.CopyInformation(mask_img)
        #sitk.WriteImage(MidAntRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\MidAntR.nii.gz')

        MidAntLMask = sitk.GetImageFromArray(MidAntLArr)
        MidAntLMask.CopyInformation(mask_img)
        #sitk.WriteImage(MidAntLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\MidAntL.nii.gz')

        MidPostRMask = sitk.GetImageFromArray(MidPostRArr)
        MidPostRMask.CopyInformation(mask_img)
        #sitk.WriteImage(MidPostRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\MidPostR.nii.gz')

        MidPostLMask = sitk.GetImageFromArray(MidPostLArr)
        MidPostLMask.CopyInformation(mask_img)
        #sitk.WriteImage(MidPostLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\MidPostL.nii.gz')

        ## Base
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        UpperArrNZ = UpperArr.nonzero()
        UpCenterX = int(min(UpperArrNZ[1] + round((max(UpperArrNZ[1]) - min(UpperArrNZ[1])) / 2)))
        UpCenterY = int(min(UpperArrNZ[2] + round((max(UpperArrNZ[2]) - min(UpperArrNZ[2])) / 2)))

        # Create blank masks/arrays
        UpAntRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        UpAntRArr = sitk.GetArrayFromImage(UpAntRMask)
        UpAntLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        UpAntLArr = sitk.GetArrayFromImage(UpAntLMask)
        UpPostRMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        UpPostRArr = sitk.GetArrayFromImage(UpPostRMask)
        UpPostLMask = sitk.Image(512, 512, 26, sitk.sitkInt8)
        UpPostLArr = sitk.GetArrayFromImage(UpPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(UpperArrNZ[2])):
            if UpperArrNZ[1][index] < UpCenterX:  # if anterior
                if UpperArrNZ[2][index] < UpCenterY:  # if right anterior
                    UpAntRArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
                else:  # else left anterior
                    UpAntLArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
            elif UpperArrNZ[2][index] < UpCenterY:  # else posterior, if right posterior
                UpPostRArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
            else:  # else left posterior
                UpPostLArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        UpAntRMask = sitk.GetImageFromArray(UpAntRArr)
        UpAntRMask.CopyInformation(mask_img)
        #sitk.WriteImage(UpAntRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\UpAntR.nii.gz')

        UpAntLMask = sitk.GetImageFromArray(UpAntLArr)
        UpAntLMask.CopyInformation(mask_img)
        #sitk.WriteImage(UpAntLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\UpAntL.nii.gz')

        UpPostRMask = sitk.GetImageFromArray(UpPostRArr)
        UpPostRMask.CopyInformation(mask_img)
        #sitk.WriteImage(UpPostRMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\UpPostR.nii.gz')

        UpPostLMask = sitk.GetImageFromArray(UpPostLArr)
        UpPostLMask.CopyInformation(mask_img)
        #sitk.WriteImage(UpPostLMask, r'T:\MIP\Katie_Merriman\RadiomicsProject\UpPostL.nii.gz')



        # for each segment,
        seg_data.append(self.calculateRadiomics(patient, "ApexAntR", LowAntRMask))
        seg_data.append(self.calculateRadiomics(patient, "ApexAntL", LowAntLMask))
        seg_data.append(self.calculateRadiomics(patient, "ApexPostR", LowPostRMask))
        seg_data.append(self.calculateRadiomics(patient, "ApexPostL", LowPostLMask))
        seg_data.append(self.calculateRadiomics(patient, "MidAntR", MidAntRMask))
        seg_data.append(self.calculateRadiomics(patient, "MidAntL", MidAntLMask))
        seg_data.append(self.calculateRadiomics(patient, "MidPostR", MidPostRMask))
        seg_data.append(self.calculateRadiomics(patient, "MidPostL", MidPostLMask))
        seg_data.append(self.calculateRadiomics(patient, "BaseAntR", UpAntRMask))
        seg_data.append(self.calculateRadiomics(patient, "BaseAntL", UpAntLMask))
        seg_data.append(self.calculateRadiomics(patient, "BasePostR", UpPostRMask))
        seg_data.append(self.calculateRadiomics(patient, "BasePostL", UpPostLMask))

        ## calculate single section minimums, maximums, and difference from whole prostate data

        ##################FIGURE OUT HOW VOLUME PLAYS INTO THIS!!! IF PROST VOL FIRST ELEMENT, SEG VOL FIRST ELEMENT!####
        minSegList = []
        maxSegList = []
        diffSegList = []
        for segs in range(12):
            for features in range(len(seg_data[1])-1):
                if segs == 1:
                    minSegList.append(seg_data[1][features+1]) # save radiomics from element [1] on into every other spot of min list
                    minSegList.append(seg_data[1][0]) # save name of first segment to right of each value
                    maxSegList.append(seg_data[1][features+1]) # save radiomics from element [1] on into every other spot of min list
                    maxSegList.append(seg_data[1][0]) # save name of first segment to right of each value
                    diffSegList.append(seg_data[1][features+1] - prost_data[features+1])
                        # save difference in radiomics between each single segment element and whole prostate element
                                 # from element [1] on into every other spot of min list
                    diffSegList.append(seg_data[1][0]) # save name of first segment to right of each value
                else:
                    if seg_data[segs][features+1]<minSegList[features*2]:
                        minSegList[features * 2] = seg_data[segs][features+1]  # update list with new minimum
                        minSegList[features * 2 + 1] = seg_data[segs][0]  # update segment associated with new minimum
                    if seg_data[segs][features + 1] > minSegList[features * 2]:
                        maxSegList[features * 2] = seg_data[segs][features + 1]  # update list with new maximum
                        maxSegList[features * 2 + 1] = seg_data[segs][0]  # update segment associated with new maximum
                    if (seg_data[segs][features + 1]- prost_data[features + 2]) > diffSegList[features*2]:
                        diffSegList[features * 2] = seg_data[segs][features + 1] - prost_data[features + 1]
                        # save difference in radiomics between each single segment element and whole prostate element
                        # from element [1] on into every other spot of min list
                        diffSegList[features * 2 + 1] = seg_data[segs][0]  # save name of first segment to right of each value


        segment_data = seg_data[0] + seg_data[1] + seg_data[2] + seg_data[3] + seg_data[4] + seg_data[5] + seg_data[6] \
                       + seg_data[7] + seg_data[8] + seg_data[9] + seg_data[10] + seg_data[11] + minSegList + maxSegList\
                       + diffSegList
        return segment_data


    def calculateLesion(self, patient,voi_list,prost_data):
        lesion_data = []
        for lesion in voi_list:
            voiMask = voi_list[lesion][0]  # path join patient path + mrn_with_date + wp_bt.nii.gz
            mask_img = sitk.ReadImage(voiMask)
            lesion_data.append(self.calculateRadiomics(patient, voi_list[lesion][1],mask_img))

        return lesion_data

    def calculateRadiomics(self, patient, name, mask):
        rad = [name]
        if mask:
            for num in range(len(patient)):
                rad.append(patient[num])



        """

        settings = {}
        settings['geometryTolerance'] = 0.0001
        rad_info = [patient_id, file, pirads]

        # load t2 and adc niftis, create arrays
        t2_img_path = os.path.join(self.patientFolder, patient_id, 'T2n.nii')
        t2_img = sitk.ReadImage(t2_img_path)

        adc_img_path = os.path.join(self.patientFolder, patient_id, 'ADCn.nii')
        adc_img = sitk.ReadImage(adc_img_path)

        highB_img_path = os.path.join(self.patientFolder, patient_id, 'highBn.nii')
        highB_img = sitk.ReadImage(highB_img_path)

        mask_img = mask

        applyLog = False
        applyWavelet = False



        ### Get T2 values
        mask_vol = 0.001 * radiomics.shape.RadiomicsShape(t2_img, mask_img).getMeshVolumeFeatureValue()
        mask_SA = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSurfaceAreaFeatureValue()
        mask_sphr = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSphericityFeatureValue()
        mask_SVR = radiomics.shape.RadiomicsShape(t2_img, mask_img).getSurfaceVolumeRatioFeatureValue()
        mask_3Dd = radiomics.shape.RadiomicsShape(t2_img, mask_img).getMaximum3DDiameterFeatureValue()
        mask_elong = radiomics.shape.RadiomicsShape(t2_img, mask_img).getElongationFeatureValue()
        mask_flat = radiomics.shape.RadiomicsShape(t2_img, mask_img).getFlatnessFeatureValue()
        rad_info.extend([mask_vol, mask_SA, mask_sphr, mask_SVR, mask_3Dd, mask_elong, mask_flat])

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(t2_img, mask_img)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)


        results = firstOrderFeatures.execute()
        for (key, val) in six.iteritems(results):
            rad_info.append(float(val))


        firstOrderFeatures = firstorder.RadiomicsFirstOrder(adc_img, mask_img, **settings)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('10Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)
        firstOrderFeatures.enableFeatureByName('Variance', True)
        #firstOrderFeatures.enableAllFeatures()

        # print('Calculating first order features...')
        #settings = {}
        #settings['geometryTolerance'] = 0.00001
        #results = firstOrderFeatures.execute(**settings)
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            #print('  ', key, ':', val)
            rad_info.append(float(val))



        firstOrderFeatures = firstorder.RadiomicsFirstOrder(highB_img, mask_img, **settings)
        firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableFeatureByName('Median', True)
        firstOrderFeatures.enableFeatureByName('90Percentile', True)
        firstOrderFeatures.enableFeatureByName('InterquartileRange', True)
        firstOrderFeatures.enableFeatureByName('Entropy', True)
        firstOrderFeatures.enableFeatureByName('Variance', True)
        firstOrderFeatures.enableFeatureByName('Uniformity', True)
        firstOrderFeatures.enableFeatureByName('Variance', True)

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            #print('  ', key, ':', val)
            rad_info.append(float(val))


        # THIS DID NOT WORK.  Got error "AttributeError: 'RadiomicsGLCM' object has no attribute 'enableFeaturesByName'"
        #GLCMextractor = glcm.RadiomicsGLCM(t2_img, mask_img)
        #GLCMextractor.enableFeaturesByName('Idm',True)
        #GLCMextractor.enableFeaturesByName('Idmn',True)
        #GLCMextractor.enableFeaturesByName('Id',True)
        #GLCMextractor.enableFeaturesByName('Idn',True)
        # print('Calculating first order features...')
        #results = GLCMextractor.execute()
        #btest = []
        # print('Calculated first order features: ')
        #for (key, val) in six.iteritems(results):
        #    print('  ', key, ':', val)
        #    btest.append(float(val))

        # THIS DID NOT WORK EITHER. Gives "KeyError: 'pxSuby'"
        #t2IDM = radiomics.glcm.RadiomicsGLCM(t2_img, mask_img).getIdmFeatureValue()
        #adcIDM = radiomics.glcm.RadiomicsGLCM(adc_img, mask_img).getIdmFeatureValue()
        #highbIDM = radiomics.glcm.RadiomicsGLCM(highB_img, mask_img).getIdmFeatureValue()
        # etc...


        #THIS WORKS BUT RESULTS HAVE TO BE MANIPULATED STRANGELY. See note at end
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(glcm=['Idm', 'Idmn', 'Id', 'Idn'], glszm=['GrayLevelNonUniformity','GrayLevelNonUniformityNormalized', 'ZoneEntropy', 'ZonePercentage'], ngtdm=['Coarseness'])
        #extractor._setTolerance(0.000002)


        t2_results = []
        results = extractor.execute(t2_img, mask_img)
        for (key, val) in six.iteritems(results):
            t2_results.append(val)

        adc_results = []
        results = extractor.execute(adc_img, mask_img)
        for (key, val) in six.iteritems(results):
            adc_results.append(val)

        highB_results = []
        results = extractor.execute(highB_img, mask_img)
        for (key, val) in six.iteritems(results):
            highB_results.append(val)

        # Note to future self: this is a hackey way to deal with the odd output format from the extractor.
        # The first 22 key/val pairs are info on the pyradiomics package, details of the input images, etc
        # Then the actual results are saved as 0-dimensional arrays, which create errors when you try to access the one
        #    value stored in the array (i.e. highB_results[22] gives error, as does highB_results[22][0]).  However,
        #    because there is only one value, min and max are the same, and accessible.  Gives actual desired value.


        rad_info.extend([t2_results[22].min(), adc_results[22].min(), highB_results[22].min(),
                         t2_results[23].min(), adc_results[23].min(), highB_results[23].min(),
                         t2_results[24].min(), adc_results[24].min(), highB_results[24].min(),
                         t2_results[25].min(), adc_results[25].min(), highB_results[25].min(),
                         t2_results[26].min(), adc_results[26].min(), highB_results[26].min(),
                         t2_results[27].min(), adc_results[27].min(), highB_results[27].min(),
                         t2_results[28].min(), adc_results[28].min(), highB_results[28].min(),
                         t2_results[29].min(), adc_results[29].min(), highB_results[29].min(),
                         t2_results[30].min(), adc_results[30].min(), highB_results[30].min()])

                        #'t2IDM', 'adcIDM', 'highb
        """

        return rad


if __name__ == '__main__':
    c = featureCalculator()
    c.calculate()
    print('Conversions complete')