### Script for some analysis of the MIMIC data

#### Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#### Read in the data
chart_events = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv', sep= ',', parse_dates=['charttime'])
event_index = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv')

#### Pull out time series of just stuff we are interested
#Find which labels occur in chartevents
chart_events = chart_events.merge(event_index,on=['itemid'])
unique_labels = chart_events['label'].unique()

#Columns to merge:
temp = [0, 297, 441]
tempc =  [133, 296, 440]
peak_insp = [1, 195, 657]
RR = [2, 55, 81, 88, 89, 498, 557, 729, 730]
sysBP = [4, 108, 124, 169, 172]
# art mean? 744
diaBP = [5, 109, 125, 170, 171, 913]
sats =[17, 58, 294] 
fiO2 = [48, 452, 722, 64]
flowO2 = [196, 446, 453, 574, 658]
#53 tital vol 68 80 87 468 501
#54 minute vol
#56 Mean airway pressure
gluc = [76, 116, 368]
CVP = [129, 410]
#195 bipap ipap 
#208 venous o2 130
dialysis = [227, 226, 228, 230, 511, 512, 1508] 
GCS = [320, [1104, 1105, 1106], [1513, 1514, 1515]]
#325 avpu
ventilation = [525, 470, 469, 919, 1162]
#727 sedation score

#Merge same fields
chart_events['combinations'] = chart_events['label']
chart_events['comb_values'] = chart_events['valuenum']

#Merge tempF
chart_events[chart_events['label'].isin(unique_labels[[temp + tempc]])]['combinations'] = 'temp'

#Convert C to F and merge
tempf = chart_events[chart_events['label'].isin(unique_labels[tempc])]['comb_values']*(9/5) + 32
chart_events[chart_events['label'].isin(unique_labels[tempc])]['comb_values'] = tempf

#Merge others
chart_events[chart_events['label'].isin(unique_labels[RR])]['combinations'] = 'RR'
chart_events[chart_events['label'].isin(unique_labels[sysBP])]['combinations'] = 'sysBP'
chart_events[chart_events['label'].isin(unique_labels[diaBP])]['combinations'] = 'diaBP'
chart_events[chart_events['label'].isin(unique_labels[sats])]['combinations'] = 'sats'
chart_events[chart_events['label'].isin(unique_labels[fiO2])]['combinations'] = 'fiO2'
chart_events[chart_events['label'].isin(unique_labels[flowO2])]['combinations'] = 'flowO2'




#Include only numeric values
chart_events = chart_events[chart_events['value'].astype(str).str.isdigit()]

#First split up table by patient
patients = chart_events['subject_id'].unique()

##Work through all individual patients, make into nested list of dfs for each var
#First work through unique patients
patient_dfs = list()
for i in range(patients.shape[0]):

    #Next pull out unique variables
    unique_vars = chart_events[chart_events['subject_id'] == patients[i]]['label'].unique()
    vars_list = list()

    #Temp df containing only this patients vars
    pt_chart = chart_events[(chart_events['subject_id'] == patients[i])]

    #Pull out number of times each appears and sort them
    length_vars = np.zeros(unique_vars.shape[0])

    #Number of times each one appears
    for j in range(unique_vars.shape[0]):
        length_vars[j] = pt_chart[pt_chart['label'] == unique_vars[j]].shape[0]

    #Sort and save sorted bits as new unique vars which we can save as first list item
    unique_vars = unique_vars[np.argsort(length_vars)[::-1]]
    vars_list.append(unique_vars)

    #Now work through and subset them
    for j in range(unique_vars.shape[0]):

        #Now work through vars
        vars_list.append(chart_events[(chart_events['subject_id'] == patients[i]) & (chart_events['label'] == unique_vars[j])])

    patient_dfs.append(vars_list)



##### Do some plotting of variables

plt.figure(figsize=(16,5), dpi=300)
plt.plot(patient_dfs[0][0]['charttime'], patient_dfs[0][0]['valuenum'], color='tab:red')
fig.savefig('Project/PICU_project/figs/multiple_temp.png')
plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
plt.show()

#Pull out only the ones that occur multiple times
lengths = np.zeros(unique_labels.shape[0])
for i in range(len(unique_labels)):
    lengths[i] = chart_events[chart_events['label'] == unique_labels[i]].shape[0]

common_unique_labels = unique_labels[lengths > 1000]

#Remove all alarm labels
alarms = np.array([])
for i in range(len(unique_labels)):
    x = re.match('.*Alarm.*', unique_labels[i], re.IGNORECASE)

    if x:
        print('nothing')
    else: 
        np.append(alarms, i)


#Pull out only labels which occur in all patients
pt_label_combos = chart_events.drop_duplicates(subset = ["label", 'subject_id'])[["label", 'subject_id']]

#Labels occurring in first patient
overlapping_labels = list()
for i in range(len(patients)):

    overlapping_labels.append(chart_events[chart_events['subject_id'] == patients[i]]['label'].unique())


#Here make a function where you can choose a parameter from inside the uniqe variables from index
def plot_multiple_patients(df, label, output_location, title="", xlabel='Date', ylabel='Value', dpi=500):
    ''' Function to do some plotting of timeseries data, plots multiple different patients for parameter
    Takes parameters df (pandas dataframe), x and y values, plots them
    You have to choose a label to print
    Uses the input from the index
    '''

    #Get location of label in unique labels (this corresponds to the location in the nested list)
    loc = np.where(unique_labels == label)[0][0]

    #Instantiate figure 
    fig, axs = plt.subplots(4, 4, figsize=(15,15))
    
    #Work through the differnt patients
    for i in range(4):
        for j in range(4):

            x = df[4*i + j][loc]['charttime']
            y = df[4*i + j][loc]['valuenum']
            axs[i, j].plot(x, y)
            axs[i, j].set_title('Patient ' + str(4*i + j + 1))


    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/multiple_' + output_location + '.png')


for i in range(10):
    plot_multiple_patients(patient_dfs, unique_labels[i], unique_labels[i], unique_labels[i])    



def plot_multiple_params(df, patient, labels = labels, title="", xlabel='Date', ylabel='Value', dpi=500):
    ''' Function to do some plotting of timeseries data, plots multiple parameters for same patient
    Takes parameters df (pandas dataframe), x and y values, plots them
    You have to choose a label to print
    Uses the input from the index
    Note this function doesn't work atm - not sure why
    '''

for patient in range(16):
    #Instantiate figure 
    fig, axs = plt.subplots(4, 4, figsize=(15,15), sharex=True)
    
    #Work through the different patients
    for i in range(4):
        for j in range(4):
            
            #First object in list for each pt is list of vars
            x = df[patient][4*i + j + 1]['charttime']
            y = df[patient][4*i + j + 1]['valuenum']
            axs[i, j].plot(x, y)
            axs[i, j].set_title(df[patient][0][4*i + j])

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/multiple_params' + str(patient) + '.png')

#Print out all labels
for i, j in enumerate(unique_labels[0:1000]):
    print(i, j)

#Labels we want
labels = unique_labels[[0, 1,  2, 3, 5, 6, 7, 9, 10, 13, 14, 17, 19, 20, 25, 35]]


#Make all the figs
for i in range(15):
    plot_multiple_params(patient_dfs, i, labels, title = 'Patient ' + str(i))






#### Do some clustering
#First pull out the average rank of the different parameters

positions = np.array(len(unique_labels))

for i in range(len(patients)):
    for j in range(len(unique_labels)):
        if pd.DataFrame([unique_labels[j]]).isin(patient_dfs[i][0])