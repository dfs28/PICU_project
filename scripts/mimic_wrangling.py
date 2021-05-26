### Script for some analysis of the MIMIC data

#### Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#### Read in the data
chart_events = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv', sep= ',', parse_dates=['charttime'])
event_index = pd.read_csv('Project/PICU_project/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv')

#### Pull out time series of just stuff we are interested
#Find which labels occur in chartevents
chart_events = chart_events.merge(event_index,on=['itemid'])
unique_labels = chart_events['label'].unique()

#Include only numeric values
chart_events = chart_events[chart_events['value'].astype(str).str.isdigit()]

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
fig.savefig('Project/PICU_project/figs/multiple_temp.png')
plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
plt.show()

#Pull out only the ones that occur multiple times
lengths = np.zeros(unique_labels.shape[0])
for i in range(len(unique_labels)):
    lengths[i] = chart_events[chart_events['label'] == unique_labels[i]].shape[0]




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
    '''

    #Instantiate figure 
    fig, axs = plt.subplots(4, 4, figsize=(15,15))
    
    #Work through the different patients
    for i in range(4):
        for j in range(4):
            
            loc = np.where(unique_labels == labels[4*i + j])[0][0]
            x = df[patient][loc]['charttime']
            y = df[patient][loc]['valuenum']
            axs[i, j].plot(x, y)
            axs[i, j].set_title(labels[4*i + j])

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/multiple_params' + str(patient) + '.png')

#Print out all labels
for i, j in enumerate(unique_labels):
    print(i, j)

#Labels we want
labels = unique_labels[[0, 1,  2, 3, 4, 5, 17, 76, 80, 81, 108, 109, 124, 125, 722,  727]]

for i in range(15):
    plot_multiple_params(patient_dfs, i, labels, title = 'Patient ' + str(i))

