import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from Run.CreateGraph.CreateMyGraph import load_txt
from options import parse_args

def process_patient(Patient, LabelDataPath, CellTypes):
    TXTData = load_txt(LabelDataPath, Patient)
    TXTData = TXTData[TXTData['Class'].isin(CellTypes)]
    local_data_list = []

    counts = TXTData['Class'].value_counts()
    for CellType in CellTypes:
        local_data_list.append({
            'Patient': Patient,
            'CellType': CellType,
            'CellNumber': counts.get(CellType, 0)  # Use 0 if the CellType is not in the counts
        })
    return local_data_list

opt = parse_args()
FollowUpData = pd.read_csv(opt.follow_up_data, sep='\t', engine='python')
FollowUpData.dropna(axis=0, how='all', inplace=True)

CellTypes = ['Tumor cells',
             'Vascular endothelial cells',
             'Lymphocytes',
             'Fibroblasts',
             'Biliary epithelial cells',
             'Hepatocytes',
             'Other']
LabelDataPath = opt.label_data_path
Patients = os.listdir('/data/yuanyz/datasets/patientminimum_spanning_tree256412/test')
# Patients = ['201325131']
# Use joblib for parallel processing
results = Parallel(n_jobs=1)(delayed(process_patient)(Patient, LabelDataPath, CellTypes) for Patient in tqdm(Patients))

# Flatten the results
data_list = [item for sublist in results for item in sublist]

data = pd.DataFrame(data_list)
data.to_csv('Cell_Number_test.csv', index=False)
