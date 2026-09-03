import pydicom
def read_dicom(file_path):
    dataset = pydicom.dcmread(file_path)
    return dataset