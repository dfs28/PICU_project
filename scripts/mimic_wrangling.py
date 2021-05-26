### Script for some analysis of the MIMIC data

#### Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#### Read in the data
chart_events = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv', sep= ',', parse_dates=['charttime'])
event_index = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv')

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



##### Do some plotting of variables

plt.figure(figsize=(16,5), dpi=300)
plt.plot(patient_dfs[0][0]['charttime'], patient_dfs[0][0]['valuenum'], color='tab:red')
plt.savefig('Project/PICU_project/figs/mimic_temp.png')
plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
plt.show()

#Find which labels occur in chartevents
chart_events = chart_events.merge(event_index,on=['itemid'])
unique_labels = chart_events['label'].unique()

#Include only numeric values
chart_events = chart_events[chart_events['value'].astype(str).str.isdigit()]

#Pull out only the ones that occur multiple times
lengths = np.zeros(unique_labels.shape[0])
for i in range(len(unique_labels)):
    lengths[i] = chart_events[chart_events['label'] == unique_labels[i]].shape[0]




#Here make a function where you can choose a parameter from inside the uniqe variables from index
def plot_df(df, x, y, output_location, title="", xlabel='Date', ylabel='Value', dpi=100):
    ''' Function to do some plotting of timeseries data
    Takes parameters df (pandas dataframe), x and y values, plots them
    Uses the input from the index
    '''
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')    
