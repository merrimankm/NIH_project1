# Hemorrhage_ConvertByStudy.py
#   Created 9/27/22 by Katie Merriman
#   Searches through designated folder for DICOM files of all patients
#   Determines DICOM orientation and series type, sorts DICOMS, converts to NIfTI, and saves to designated folder

# Requirements
#   pydicom
#   SimpleITK
#   csv
#   pandas
#   shutil

# Inputs (currently in class definition)
#   1: path to source directory with all patients' DICOMS
#           - converted files will be saved in each patient folder and csv report will be saved in top level folder
#   2: paths to "MST" folder and
#   3:          "T1_TSE_ax" folder where converted NIfTIs will be saved
#
# Note: orientation folders within series type folders within patient DICOM folders should NOT already exist
#               - eg: T:\MIP\Robert Huang\2022_06_01\other\hemorrhage\8084257\ZJHVJ0GX\3RQ1CQX3\MST_DICOMS\axial
#               - creation of new folder triggers addition of that folder to NIfTI conversion list
#               - existing orientation folders will not be flagged for NIfTI conversion

# Outputs
#   1: Sorted DICOMs saved in folders within each DICOM folder in patient folders
#   2: Converted NIfTIs with anonymized names indicating orientation
#               sorted and saved in top level "MST" and "T1_TSE_ax" folders
#   3: Csv report of corresponding MRNs and anonymized names with number of NIfTIs created for each MRN,
#               saved in top level source folder as "AnonList"
#   4: Csv report of all NIfTI files created, with corresponding MRN and file path,
#               saved in top level source folder as "convertedDICOMS.csv"


import os
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
import csv
import pandas as pd
import shutil

class DICOMconverter():
    def __init__(self):
        pathways = 1
            # I just do this so I can easily switch between the versions of the filepaths needed to debug on my computer vs run on Lambda.
            # if on lambda, I just change "pathways" to 0 instead of commenting/uncommenting multiple lines

        if pathways:
            self.patientFolder = r'T:\MIP\Robert Huang\2022_11_01\other\no hemorrhage'
            #self.patientFolder = r'T:\MIP\Robert Huang\2022_06_01\other\hemorrhage'
            self.saveMST = r'T:\MIP\Robert Huang\2022_06_01\other\hemorrhage\MST'
            self.saveT1 = r'T:\MIP\Robert Huang\2022_06_01\other\hemorrhage\T1_TSE_ax'

        else:
            self.patientFolder = 'Mdrive_mount/MIP/Robert Huang/2022_06_01/other/hemorrhage'
            self.saveMST = 'Mdrive_mount/MIP/Robert Huang/2022_06_01/other/hemorrhage/MST'
            self.saveT1 = 'Mdrive_mount/MIP/Robert Huang/2022_06_01/other/hemorrhage/T1_TSE_ax'

        self.AnonList = []
        self.Conversions = []

    def convertPatients(self):
        patient = []

        patientFolders = os.listdir(self.patientFolder)
        excludeList = ['.txt', '.csv']
        for patientID in patientFolders:
            if not any(substring in patientID for substring in excludeList):
                patientPath = os.path.join(self.patientFolder, patientID)
                patient.append([patientID, patientPath])

        for i in range(0, len(patient)):
            anonName = str(i+91)
            if len(anonName)<2:
                anonName = '00'+anonName
            elif len(anonName)<3:
                anonName = '0'+anonName
            anonName = 'SURG-'+anonName
            self.AnonList.append([patient[i][0], anonName, 0])
            self.AnonList[i][2] = self.sortDICOMS(patient[i], anonName)  # stores number of successful conversions
            # print(self.DICOMdetails)

        self.create_csv_files()
        print('conversion complete')


    def sortDICOMS(self, p, anonName):
        success = 0
        dicomFoldersList = []
        suffix_list = ['.voi', '.stl', '.xml', '.csv', '.xlsx', '.doc', '.txt', '.jpg', '.png']
        ignore_list = ['DICOMDIR', 'LOCKFILE', 'VERSION']
        ignore_folder = ['axial_DICOM', 'sagittal_DICOM', 'coronal_DICOM', 'axial', 'sagittal', 'coronal']
        small3D = ['MST', 'SmallFOV']
        print('searching ', p[0], ' for DICOMs')

        for root, dirs, files in os.walk(p[1]):
            for name in files:
                ## ignore common non-DICOM files
                if name.endswith(tuple(suffix_list)):
                    continue
                ## look for DICOMS and sort by type and orientation
                else:
                    if name not in ignore_list:
                        filePath = os.path.join(root, name)

                        dicomFolder = os.path.dirname(filePath)
                        if dicomFolder.endswith(tuple(ignore_folder)):
                            continue
                        # check if file is DICOM, get orientation and series type
                        try:
                            ds = dcmread(filePath)

                            # get orientation name
                            orthog = ds.ImageOrientationPatient
                            if abs(orthog[1]) > .5:
                                orientation = 'sagittal'
                            elif abs(orthog[4]) > .5:
                                orientation = 'axial'
                            else:
                                orientation = 'coronal'

                            # get series type
                            seriesName = ds.SeriesDescription
                            if any(x in seriesName for x in small3D):
                                seriesType = 'MST_DICOMS'
                            elif 'T1 TSE ax' in seriesName:
                                seriesType = 'T1_TSE_ax_DICOMS'
                            else:
                                continue

                        except IOError:
                            # print(f'No such file')
                            break
                        except InvalidDicomError:
                            # print(f'Invalid Dicom file')
                            break

                        # create a new orientation-based folder within DICOM folder
                        seriesPath = os.path.join(dicomFolder, seriesType)
                        if not os.path.exists(seriesPath):
                            try:
                                os.mkdir(seriesPath)
                            except FileNotFoundError:
                                print(f'Error! Invalid path!')
                        orthogPath = os.path.join(seriesPath, orientation)
                        if not os.path.exists(orthogPath):
                            try:
                                os.mkdir(orthogPath)
                                # add folder to list for NIfTI conversion
                                dicomFoldersList.append([orthogPath, p[1], p[0], anonName, seriesType, orientation])  # add to list of DICOM folders for later convertion to NIfTI
                            except FileNotFoundError:
                                print(f'Error! Invalid path!')


                        # copy DICOM file to orientation-based folder
                        try :
                            shutil.copy(filePath, orthogPath)
                        except AttributeError:
                            continue

        if dicomFoldersList:
            print('converting files')
            success = self.DICOMconvert(dicomFoldersList)

        return success


    def DICOMconvert(self, dicomFoldersList):
        success = 0
        reader = sitk.ImageSeriesReader()
        for folder in dicomFoldersList:
            print(f'Converting', folder[2], folder[4], folder[5], ' DICOM to NIfTI...')  # let user know code is running
            # read, convert, and save NIfTIs
            f = folder[0]
            dicom_names = reader.GetGDCMSeriesFileNames(f)
            reader.SetFileNames(dicom_names)
            if 'MST' in folder[4]:
                imageFile = os.path.join(self.saveMST, folder[3] + '_' + folder[5] + '.nii.gz')
            else:
                imageFile = os.path.join(self.saveT1, folder[3] + '_' + folder[5] + '.nii.gz')
            image = reader.Execute()
            sitk.WriteImage(image, imageFile)
            success = success + 1
            self.Conversions.append([folder[2], folder[3] + '_' + folder[5]+'.nii', imageFile])

        return success


    def create_csv_files(self):
        nifti_cvsFileName = os.path.join(self.patientFolder, 'convertedDICOMS.csv')
        niftiHeader = ['MRN', 'NIfTIfile', 'NIfTIpath']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.Conversions)

        nifti_cvsFileName = os.path.join(self.patientFolder, 'AnonList2.csv')
        niftiHeader = ['MRN', 'AnonName', 'numNIfTIs']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.AnonList)

if __name__ == '__main__':
    c = DICOMconverter()
    c.convertPatients()


