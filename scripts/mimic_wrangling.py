### Script for some analysis of the MIMIC data

#### Setup
import pandas as pd
import numpy as np




#### Read in the data
chart_events = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv', sep= ',')
index = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv')

#### Pull out time series of just stuff we are interested
#First split up table by patient
patients = chart_events['subject_id'].unique()

##Work through all individual patients, make into nested list of dfs for each var
#First work through unique patients
patient_dfs = list()
for i in range(patients.shape[0]):
    
    #Next pull out unique variables
    unique_vars = chart_events[chart_events['subject_id'] == patients[i]]['itemid'].unique()
    vars_list = list()

    for j in range(unique_vars.shape[0]):
        
        #Now work through vars
        vars_list.append(chart_events[(chart_events['subject_id'] == patients[i]) & (chart_events['itemid'] == unique_vars[j])])
        
    patient_dfs.append(vars_list)



patient_dfs = chart_events[chart_events['subject_id'] == patients[1]]

itemids = chart_events['itemid'].unique()
