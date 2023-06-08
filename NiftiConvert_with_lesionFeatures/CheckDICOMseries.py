import os
import sys
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
import csv
import glob
import pandas as pd
def main(argv):
    # example usage: python csvDicom_searchConvert.py "/path/cvsFileName.csv" "/path/to/folder/of/patient/folders"
    #       "/path/to/save/folder"


    #if len(argv) < 4:
    #    print(f"Usage: {argv[0]} csvFile topFolder saveFolder", file=sys.stderr)
    #    exit(1)

    # print(f'Program started...')
    #csvFile_withPath = argv[1]
    #patientFolder = argv[2]
    #saveFolder = argv[3]
    csvFile_withPath = r"T:\MIP\Katie_Merriman\Project2Data\Patient_list_directories_short.csv"
    patientFolder = r"T:\MRIClinical\surgery_cases"
    saveFolder = r"T:\MIP\Katie_Merriman\Project2Data"

    rows = []
    nifti_csv = []
    noDCM_csv = []
    dcm_patients_csv = []

    # with open(csvFile_withPath) as file:
    #    csvreader = csv.reader(file)
    #    header = next(csvreader)
    #    for row in csvreader:
    #        rows.append(row)
    # for patient in rows:
    file = open(os.path.join(saveFolder, "DICOM_infobyseries.csv"), 'a+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows([["MRN", "dicomFoldersList", "dicomSeriesList", "dicomProtocolList", "patientPath", "savePath"]])
    file.close()

    file = open(os.path.join(saveFolder, "DICOM_info.csv"), 'a+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows([["MRN", "Patientpath", "T2path", "T2series", "ADCpath", "ADCseries", "highBpath", "highBseries",
                         "DCEpath", "DCEseries", "TotalSeries", "unknownpath"]])
    file.close()

    df_csv = pd.read_csv(csvFile_withPath, sep=',', header=0)
    for rows, file_i in df_csv.iterrows():
        patient = str(file_i['MRN'])
        if '_' in patient:
            MRNindex = patient.index('_')
            patient = str(int(patient[0:MRNindex]))
        else:
            patient = str(int(patient))


        if patient:
            if len(patient) < 7:
                patient = '0' + patient
            patient_string = patient + '*'
            patientPath = glob.glob(os.path.join(patientFolder, patient_string))
        if patientPath:
            patientPath = patientPath[0]
            patientID = os.path.basename(patientPath)
            print(f'Starting DICOM conversion for ' + patientID + "...")
            niftiInfo = search_and_convert_dicom(patientPath, saveFolder, patientID)
            print(niftiInfo)
#            for series in niftiInfo:
#                nifti_csv.append(series)
#            dcm_patients_csv.append([patientID])
        else:
#            noDCM_csv.append([patient])
            niftiInfo = [patientID, "Patient Path Error"]

        DCMseriesdata = []
        for series in range(len(niftiInfo[1])):
            DCMseriesdata.append([niftiInfo[0], niftiInfo[4], niftiInfo[1][series], niftiInfo[2][series], niftiInfo[3][series]])

        print(DCMseriesdata)

        file = open(os.path.join(saveFolder,"DICOM_infobyseries.csv"), 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([DCMseriesdata])
        file.close()

        numSeries = 0
        DCMdata = []

        DCMdata = [niftiInfo[0], niftiInfo[4]]
        numSeries = len(niftiInfo[1])
        if 'T2' in niftiInfo[3]:
            series = niftiInfo[3].index('T2')
            DCMdata = DCMdata + [niftiInfo[1][series], niftiInfo[2][series]]
        else:
            DCMdata = DCMdata + ["no T2", "no T2"]
        if 'ADC' in niftiInfo[3]:
            series = niftiInfo[3].index('ADC')
            DCMdata = DCMdata + [niftiInfo[1][series], niftiInfo[2][series]]
        else:
            DCMdata = DCMdata + ["no ADC", "no ADC"]
        if 'highB' in niftiInfo[3]:
            series = niftiInfo[3].index('highB')
            DCMdata = DCMdata + [niftiInfo[1][series], niftiInfo[2][series]]
        else:
            DCMdata = DCMdata + ["no highB", "no highB"]
        if 'DCE' in niftiInfo[3]:
            series = niftiInfo[3].index('DCE')
            DCMdata = DCMdata + [niftiInfo[1][series], niftiInfo[2][series]]
        else:
            DCMdata = DCMdata + ["no DCE", "no DCE"]

        DCMdata = DCMdata + [str(numSeries)]

        if 'unknown' in niftiInfo[3]:
            series = niftiInfo[3].index('unknown')
            DCMdata = DCMdata + [niftiInfo[1][series]]

        print(DCMdata)

        file = open(os.path.join(saveFolder, "DICOM_info.csv"), 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([DCMdata])
        file.close()

def search_and_convert_dicom(patientPath, savePath, MRN):
    print(f'Searching for DICOM files...')

    dicomFoldersList = []
    dicomSeriesList = []
    dicomProtocolList = []

    for root, dirs, files in os.walk(patientPath):
        for name in files:
            filePath = os.path.join(root, name)
            try:
                ds = dcmread(filePath)
            except IOError:
                # print(f'No such file')
                continue
            except InvalidDicomError:
                # print(f'Invalid Dicom file')
                continue
            if name != ("DICOMDIR" or "LOCKFILE" or "VERSION"):
                dicomString = filePath[:-(len(name) + 1)]
                if dicomString not in dicomFoldersList:
                    print(dicomString)
                    if 'delete' in dicomString:
                        break
                    dicomSeries = ds.ProtocolName.replace('/', '-')
                    dicomSeries = dicomSeries.replace(" ", "_")
                    # print(f'DICOM found...')
                    ADClist = ['Apparent Diffusion Coefficient', 'adc', 'ADC', 'dWIP', 'dSSh', 'dReg']
                    if ('T2' in dicomSeries or 't2' in dicomSeries):
                        dicomSeriesType = 'T2'
                    elif (any([substring in dicomString for substring in ADClist])) or (any([substring in dicomSeries for substring in ADClist])):
                        dicomSeriesType = 'ADC'
                    else:
                        series_descript = ds.SeriesDescription
                        if any([substring in series_descript for substring in ADClist]) or (dicomString.endswith('ADC') or dicomString.endswith('adc')):
                            dicomSeriesType = 'ADC'
                        elif ('T2' in series_descript or 't2' in series_descript) or (dicomString.endswith('T2') or dicomString.endswith('t2')):
                            dicomSeriesType = 'T2'
                        elif (dicomString.endswith('DCE') or dicomString.endswith('dce')):
                            dicomSeriesType = 'DCE'
                        else:
                            dicomSeriesType = 'unknown'
                    dicomFoldersList.append(dicomString)
                    dicomSeriesList.append(dicomSeries)
                    dicomProtocolList.append(dicomSeriesType)
                    # print(dicomSeries, dicomSeriesType)

    if dicomFoldersList:
        niftiInfo = [MRN, dicomFoldersList, dicomSeriesList, dicomProtocolList, patientPath]
    else:
        niftiInfo = [MRN, 'No Dicoms Found']
    return niftiInfo




if __name__ == '__main__':
    main(sys.argv)
