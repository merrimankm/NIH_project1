# VOI_convert_w_lesion_features

# Requirements
#   SimpleITK
#   pandas
#   scikit-image
#   numpy
#   pidicom
#   dicom2nifti
#


import SimpleITK as sitk # version 2.0.2
# import SimpleITK as sitk
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
import glob
import radiomics
from radiomics import featureextractor, imageoperations, firstorder
import six
np.set_printoptions(threshold=np.inf)


class convert_nifti():
    def __init__(self):
        #self.dicom_folder = r'T:\MRIClinical\surgery_cases'
        #self.csv_file = r'T:\MIP\Katie_Merriman\ZoeKatieCollab\VOItest.csv'
        #self.save_folder = r'T:\MIP\Katie_Merriman\ZoeKatieCollab\ConvertedDICOM'
        self.dicom_folder = 'Mdrive_mount/MRIClinical/surgery_cases'
        self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/ZoeKatieCollab/VOItest.csv'
        self.save_folder = 'Mdrive_mount/MIP/Katie_Merriman/ZoeKatieCollab/ConvertedDICOM'
        self.patient_data = []
        self.lesion_data = []
        self.error_data = []

    def create_masks_all_patients(self):
        '''
        create masks for all VOIs for all patients, save as .nii files and collect patient radiomics data from masks
        '''

        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        voi_csv = []
        # steps across patients
        for index, file_i in df_csv.iterrows():
            patient_id = str(file_i['MRN'])
            ## check if nifti folder exists
            #       missing nifti folder indicates missing converted DICOM files from csvDicom_searchConvert.py
            if not os.path.exists(os.path.join(self.save_folder, patient_id)):
                print('No Nifti folder!')
                self.error_data.append([patient_id, 'No Nifti Folder'])
            else:
                voi_list = []
                bt_voi_list = []
                ## look for VOI files in patient folder
                for root, dirs, files in os.walk(os.path.join(self.dicom_folder, patient_id)):
                    for name in files:
                        if name.endswith('.voi'):
                            filePath = os.path.join(root, name)
                            voi_list.append(filePath)
                ## sort out VOIs created by Dr. Baris Turkbey
                for f in voi_list:
                    if f.endswith('bt.voi'):
                        voi_csv.append([patient_id, f, 'Y'])
                        bt_voi_list.append(f)
                    else:
                        voi_csv.append([patient_id, f, 'N'])

                ### Check if masks are whole prostate or lesions, calculate relative volumes and min ADC
                i = 0
                PIRADSnames = ['PIRADS', 'pirads', 'PZ', 'TZ']
                PIRADS5names = ['PIRADS_5', 'pirads_5', 'PZ_5', 'TZ_5']
                PIRADS4names = ['PIRADS_4', 'pirads_4', 'PZ_4', 'TZ_4']
                pirads5 = 0
                pirads4 = 0
                pirads3 = 0
                if bt_voi_list:
                    print(f'    Converting VOI files for', patient_id)
                    for file in bt_voi_list:
                        voi_name = os.path.basename(file)
                        voi_name = str.replace(voi_name, '.voi', '')
                        if 'gg' in voi_name:
                            # ignore - voi is a copy of another voi within bt_voi_list
                            continue
                        [ADCmean, mask_vol, rad_info] = \
                            self.create_nifti_mask(pt_num=patient_id, patient_id=patient_id, file=file)
                        if ADCmean:
                            if any([substring in voi_name for substring in PIRADSnames]):
                                if i==0:
                                    ADCmin = ADCmean
                                    ADCmin2 = rad_info[2]
                                    ADCfile = voi_name
                                    ADCfile2 = voi_name
                                    i += 1
                                else:
                                    if ADCmean < ADCmin:
                                        ADCmin = ADCmean
                                        ADCfile = voi_name
                                    if rad_info[2] < ADCmin2:
                                        ADCmin2 = rad_info[2]
                                        ADCfile2 = voi_name
                                    i += 1
                                if any([substring in voi_name for substring in PIRADS5names]):
                                    pirads5 = pirads5 + mask_vol
                                elif any([substring in voi_name for substring in PIRADS4names]):
                                    pirads4 = pirads4 + mask_vol
                                else:
                                    pirads3 = pirads3 + mask_vol
                                self.lesion_data.append(rad_info)
                            elif 'wp' in voi_name:
                                prost = mask_vol
                    piradstotal = pirads5 + pirads4 + pirads3
                    relative_vol5 = pirads5/prost
                    relative_vol4 = pirads4/prost
                    relative_vol3 = pirads3/prost
                    relative_voltotal = piradstotal/prost
                    self.patient_data.append([patient_id, ADCmin, ADCfile, ADCmin2, ADCfile2, prost, relative_vol5,
                                              relative_vol4, relative_vol3, relative_voltotal])
                else:
                    self.error_data.append([patient_id, 'No bt.voi files found'])
                    print(f'    Could not find VOI files for', patient_id)



        voi_cvsFileName = os.path.join(self.save_folder, 'voi_list.csv')
        niftiHeader = ['MRN', 'VOI name', 'Converted?']
        with open(voi_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(voi_csv)

    def create_nifti_mask(self,pt_num='',patient_id='',file=''):
        '''
        use simple itk to read in t2 nifti, create a mask, write with same properties
        '''

        ADCmean = 0
        mask_vol = 0
        rad_info = [patient_id, file]


        #load t2 and adc niftis, create arrays
        t2_img_path = os.path.join(self.save_folder, patient_id, 'T2.nii')
        t2_img = sitk.ReadImage(t2_img_path)
        img_array = sitk.GetArrayFromImage(t2_img)
        img_array = np.swapaxes(img_array, 2, 0)

        adc_img_path = os.path.join(self.save_folder, patient_id, 'ADC.nii.gz')
        adc_img = sitk.ReadImage(adc_img_path)
        adc_array = sitk.GetArrayFromImage(adc_img)
        adc_array = np.swapaxes(adc_array, 2, 0)

        #iterate over mask and update empty array with mask
        numpy_mask = np.empty(img_array.shape)
        mask_dict = self.mask_coord_dict(patient_id=patient_id, file=file, img_shape = (img_array.shape[0], img_array.shape[1]))
        if mask_dict:
            for key in mask_dict.keys():
                numpy_mask[:,:,int(key)]=mask_dict[key]

            try:
                ## get mean ADC value
                numpyNonzero = numpy_mask.nonzero()
                #ind_test1 = numpy_mask.item(numpyNonzero[0][0], numpyNonzero[1][0], numpyNonzero[2][0])
               # ind_test2 = numpy_mask.item(numpyNonzero[0][0], numpyNonzero[1][0], numpyNonzero[2][0])
                #ADC_test1 = adc_array.item(100, 100, 20)
                ADCvalue = 0
                if len(numpyNonzero[0]):

                    for x in range(len(numpyNonzero[0])):
                       ADCvalue = ADCvalue + adc_array.item(numpyNonzero[0][x], numpyNonzero[1][x], numpyNonzero[2][x])

                    ADCmean = ADCvalue/len(numpyNonzero[0])
                else:
                    self.error_data.append([patient_id, 'No nonzero results for mask', file])
                    ADCmean = 0
            except RuntimeError:
                # No ADC for patient
                self.error_data.append([patient_id, 'No ADC for patient', file])

            numpy_mask = np.swapaxes(numpy_mask, 2, 0)
            img_out = sitk.GetImageFromArray(numpy_mask)

            #need to save as nifti
            img_out.CopyInformation(t2_img)
            for meta_elem in t2_img.GetMetaDataKeys():
                img_out.SetMetaData(meta_elem, t2_img.GetMetaData(meta_elem))
            niftiPath = os.path.basename(file)
            niftiPath = str.replace(niftiPath, '.voi', '.nii.gz')
            niftiPath = os.path.join(self.save_folder, patient_id, niftiPath)
            sitk.WriteImage(img_out, niftiPath)

            #extractor = featureextractor.RadiomicsFeatureExtractor()
            # extractor.enableAllFeatures()  # Enables all feature classes
            # Alternative: only first order
            #extractor.disableAllFeatures  # All features enabled by default
            #extractor.enableFeatureClassByName('firstorder')

            #featureVector = extractor.execute(adc_img_path, img_out)

            #for (key, val) in six.iteritems(featureVector):
            #    print("\t%s: %s" % (key, val))

            image = sitk.ReadImage(adc_img_path)
            mask = sitk.ReadImage(niftiPath)

            applyLog = False
            applyWavelet = False

            # Setting for the feature calculation.
            # Currently, resampling is disabled.
            # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
            settings = {'binWidth': 25,
                        'interpolator': sitk.sitkBSpline,
                        'resampledPixelSpacing': None}

            #
            # If enabled, resample image (resampled image is automatically cropped.
            #
            interpolator = settings.get('interpolator')
            resampledPixelSpacing = settings.get('resampledPixelSpacing')
            if interpolator is not None and resampledPixelSpacing is not None:
                image, mask = imageoperations.resampleImage(image, mask, **settings)

            #bb, correctedMask = imageoperations.checkMask(image, mask)
            #if correctedMask is not None:
            #    mask = correctedMask
            #image, mask = imageoperations.cropToTumorMask(image, mask, bb)

            #
            # Show the first order feature calculations
            #
            firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)

            firstOrderFeatures.enableFeatureByName('Mean', True)
            firstOrderFeatures.enableFeatureByName('Median', True)
            firstOrderFeatures.enableFeatureByName('10Percentile', True)
            # firstOrderFeatures.enableAllFeatures()

            # print('Will calculate the following first order features: ')
            # for f in firstOrderFeatures.enabledFeatures.keys():
                # print('  ', f)
                # print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)


            # print('Calculating first order features...')
            results = firstOrderFeatures.execute()
            # print('done')

            # print('Calculated first order features: ')
            for (key, val) in six.iteritems(results):
                # print('  ', key, ':', val)
                rad_info.append(float(val))



            ## collect radiomics shape info
            # ADCmean = radiomics.firstorder.RadiomicsFirstOrder(t2_img, img_out).getMeanFeatureValue()
            mask_vol = 0.001 * radiomics.shape.RadiomicsShape(t2_img, img_out).getMeshVolumeFeatureValue()
            mask_SVR = radiomics.shape.RadiomicsShape(t2_img, img_out).getSurfaceVolumeRatioFeatureValue()
            mask_sphr = radiomics.shape.RadiomicsShape(t2_img, img_out).getSphericityFeatureValue()
            mask_3Dd = radiomics.shape.RadiomicsShape(t2_img, img_out).getMaximum3DDiameterFeatureValue()
            mask_elong = radiomics.shape.RadiomicsShape(t2_img, img_out).getElongationFeatureValue()
            mask_flat = radiomics.shape.RadiomicsShape(t2_img, img_out).getFlatnessFeatureValue()
            rad_info.extend([mask_vol, mask_SVR, mask_sphr, mask_3Dd, mask_elong, mask_flat])
        else:
            self.error_data.append([patient_id, 'No keys in mask_dict', file])
        return [rad_info]

    def mask_coord_dict(self,patient_id='',file='',img_shape=()):
        '''
        creates a dictionary where keys are slice number and values are a mask (value 1) for area
        contained within .voi polygon segmentation
        :param patient_dir: root for directory to each patient
        :param type: types of file (wp,tz,urethra,PIRADS)
        :return: dictionary where keys are slice number, values are mask
        '''

        # define path to voi file
        voi_path=os.path.join(self.dicom_folder, patient_id, file)

        #read in .voi file as pandas df
        pd_df = pd.read_fwf(voi_path)

        # use get_ROI_slice_loc to find location of each segment
        dict=self.get_ROI_slice_loc(voi_path)

        output_dict={}
        if dict:
            for slice in dict.keys():
                values=dict[slice]
                select_val=list(range(values[1],values[2]))
                specific_part=pd_df.iloc[select_val,:]
                split_df = specific_part.join(specific_part['MIPAV VOI FILE'].str.split(' ', 1, expand=True).rename(columns={0: "X", 1: "Y"})).drop(['MIPAV VOI FILE'], axis=1)
                X_coord=np.array(split_df['X'].tolist(),dtype=float).astype(int)
                Y_coord=np.array(split_df['Y'].tolist(),dtype=float).astype(int)
                mask=self.poly2mask(vertex_row_coords=X_coord, vertex_col_coords=Y_coord, shape=img_shape)
                output_dict[slice]=mask

        return(output_dict)

    def get_ROI_slice_loc(self,path):
        '''
        selects each slice number and the location of starting coord and end coord
        :return: dict of {slice number:(tuple of start location, end location)}

        '''

        pd_df=pd.read_fwf(path)

        #get the name of the file
        filename=path.split(os.sep)[-1].split('.')[0]

        #initialize empty list and empty dictionary
        slice_num_list=[]
        last_line=[]
        loc_dict={}

        #find the location of the last line -->
        for line in range(len(pd_df)):
            line_specific=pd_df.iloc[line,:]
            as_list=line_specific.str.split(r"\t")[0]
            if "# slice number" in as_list: #find location of all #slice numbers
                slice_num_list.append(line)
            if '# unique ID of the VOI' in as_list:
                last_line.append(line)

        if len(slice_num_list) < 1:
            return None
        else:
            for i in range(len(slice_num_list)):
                # for all values except the last value
                if i<(len(slice_num_list)-1):
                    loc=slice_num_list[i]
                    line_specific=pd_df.iloc[loc,:]
                    slice_num=line_specific.str.split(r"\t")[0][0]
                    start=slice_num_list[i]+3
                    end=slice_num_list[i+1]-1
                    loc_dict.update({slice_num:(filename,start,end)})

                #for the last value
                if i == (len(slice_num_list) - 1):
                    loc = slice_num_list[i]
                    line_specific=pd_df.iloc[loc,:]
                    slice_num=line_specific.str.split(r"\t")[0][0]
                    start=slice_num_list[i]+3
                    end=(last_line[0]-1)
                loc_dict.update({slice_num: (filename, start, end)})

        return (loc_dict)

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        ''''''
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=int)
        mask[fill_row_coords, fill_col_coords] = 1
        return mask

    def create_csv_files(self):
        nifti_cvsFileName = os.path.join(self.save_folder, 'PatientDataList.csv')
        niftiHeader = ['MRN', 'Min Mean ADC', 'Lesion with Min ADC', 'Prostate Volume', 'PIRADS 5 Relative Volume', 'PIRADS 4 Relative Volume', 'PIRADS <=3 Relative Volume', 'Relative total volume']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.patient_data)

        nifti_cvsFileName = os.path.join(self.save_folder, 'VOI_ErrorsList.csv')
        niftiHeader = ['MRN', 'Error type']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.error_data)

        nifti_cvsFileName = os.path.join(self.save_folder, 'VOI_LesionsList.csv')
        niftiHeader = ['MRN', 'Lesion', 'ADC Mean', 'ADC Median', 'ADC 10th Percentile', 'ADC Entropy', 'Lesion Volume', 'Surface to Volume Ratio', 'Sphericity', 'Maximum 3D Diameter', 'Elongation', 'Flatness']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.lesion_data)
if __name__ == '__main__':
    c = convert_nifti()
    c.create_masks_all_patients()
    c.create_csv_files()
