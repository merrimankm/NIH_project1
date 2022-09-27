# Hemorrhage_checkStudy.py
#   Created 9/27/22 by Katie Merriman
#   Searches through designated folder for DICOM files of all patients
#   Saves info on series, protocol, TR, TE, spacing, size, and orientation

# Requirements
#   pydicom
#   SimpleITK
#   csv
#   pandas
#   shutil

# Inputs (currently in class definition)
#   1: path to source directory with all patients' DICOMS

# Outputs
#   1: Csv report saved in source directory

import os
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
import csv
import pandas as pd
import shutil

class Dummy:
    def __eq__(self, other):
        return True

class DICOMconverter():
    def __init__(self):
        pathways = 0
            # I just do this so I can easily switch between the versions of the filepaths needed to debug on my computer vs run on Lambda.
            # if on lambda, I just change "pathways" to 0 instead of commenting/uncommenting multiple lines

        if pathways:
            #self.csv_file = r'T:\MIP\Robert Huang\2022_06_01\other\hemorrhage\patients2convert.csv'
            self.patientFolder = r'T:\MIP\Robert Huang\2022_06_01\other\hemorrhage'

        else:
            #self.csv_file = 'Mdrive_mount/MIP/Robert Huang/2022_06_01/other/hemorrhage/patients2convert.csv'
            self.patientFolder = 'Mdrive_mount/MIP/Robert Huang/2022_06_01/other/hemorrhage'


        self.DICOMdetails = []

    def convertPatients(self):
        patient = []

        patientFolders = os.listdir(self.patientFolder)
        excludeList = ['.txt', '.csv']
        for patientID in patientFolders:
            if not any(substring in patientID for substring in excludeList):
                patientPath = os.path.join(self.patientFolder, patientID)
                patient.append([patientID, patientPath, 0])

        for i in range(0, len(patient)):
            self.sortDICOMS(patient[i]) # stores number of successful conversions
            #print(self.DICOMdetails)
            self.DICOMdetails.append([])

        self.create_csv_files()
        print('conversion complete')

    def sortDICOMS(self, p):
        dicomList = []
        suffix_list = ['.voi', '.stl', '.xml', '.csv', '.xlsx', '.doc', '.txt', '.jpg', '.png']
        ignore_list = ['DICOMDIR', 'LOCKFILE', 'VERSION']
        print('searching ', p[0], ' for DICOMs')

        for root, dirs, files in os.walk(p[1]):
            for name in files:

                ## ignore common non-DICOM files
                if name.endswith(tuple(suffix_list)):
                    continue

                ## look for DICOMS and sort by orientation
                else:
                    if name not in ignore_list:
                        filePath = os.path.join(root, name)

                        # check if file is DICOM, get orientation
                        try:
                            ds = dcmread(filePath)
                            series = ds.SeriesDescription
                            protocol = ds.ProtocolName
                            TE = ds.EchoTime
                            TR = ds.RepetitionTime
                            orthog = ds.ImageOrientationPatient
                            [spacing_x, spacing_y] = ds.PixelSpacing
                            sliceThickness = ds.SliceThickness
                            [size_x,size_y] = ds.pixel_array.shape
                            DICOMdetails = [p[0], series, protocol, TE, TR, spacing_x, spacing_y, sliceThickness, size_x, size_y]
                            detailsCHECK = [p[0], series, protocol, TE, TR, spacing_x, spacing_y, sliceThickness, size_x, size_y, Dummy(), Dummy(), Dummy()]
                            if detailsCHECK not in dicomList:
                                DICOMdetails.extend([0, 0, 0])
                                dicomList.append(DICOMdetails)
                            detailsIndex = dicomList.index(detailsCHECK)
                            if abs(orthog[1]) > .5: #sagittal
                                dicomList[detailsIndex][11] = dicomList[detailsIndex][11] + 1
                            elif abs(orthog[4]) > .5: #axial
                                dicomList[detailsIndex][10] = dicomList[detailsIndex][10] + 1
                            else: #coronal
                                dicomList[detailsIndex][12] = dicomList[detailsIndex][12] + 1

                        except IOError:
                            # print(f'No such file')
                            break
                        except InvalidDicomError:
                            # print(f'Invalid Dicom file')
                            break

        self.DICOMdetails.extend(dicomList)
        return


    def create_csv_files(self):
        nifti_cvsFileName = os.path.join(self.patientFolder, 'DICOMdetails2.csv')
        niftiHeader = ['MRN', 'series', 'protocol', 'TE', 'TR', 'spacing_x', 'spacing_y', 'sliceThickness', 'size_x', 'size_y', 'numAxial', 'numSagittal', 'numCoronal']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.DICOMdetails)


if __name__ == '__main__':
    c = DICOMconverter()
    c.convertPatients()
    c.create_csv_files()


