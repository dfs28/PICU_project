##### Script for clustering and data visualisation

## Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#Read in processed data
flowsheet = pd.read_csv('Project/Project_data/files/flowsheet_output.csv', sep= ',', parse_dates=['taken_datetime'])


#Helper function for returning the column number as a name
def return_colname(original_column, sheet, give_inputs = False):
    """
    Function to give original column as string
    """
    
    #Get the unique values flexibly
    if type(original_column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[original_column]].unique()
        
        #Make sure original column is a string
        original_column = sheet.columns[original_column]

    elif type(original_column) == str:
        unique_inputs = sheet.loc[:, (original_column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    if not give_inputs:
        return original_column
    else:
        return original_column, unique_inputs


#Split up by patient
def split_by_pt(sheet, col):
    """
    Function to split up a sheet by patient \n
    Takes a pd.DataFrame, specify column as text
    """
    patient_dfs = list()
    unique_patients = sheet[col].unique()
    
    #Work through the the patients and split into new sheets
    for i, j in enumerate(unique_patients):
        temp_df = sheet.loc[sheet[col] == j, :]
        patient_dfs.append(temp_df) 

    return patient_dfs

#Start with only numeric values for now
def return_numeric(sheet):
    """
    Function to return only the numeric columns in a sheet
    """

    #Work out which columns are numeric
    numeric_dtypes = sheet.dtypes.unique()
    sheet_dtypes = sheet.dtypes.isin(numeric_dtypes[[1, 2, 3]])
    
    #Make sure you keep the patient ID
    sheet_dtypes[0] = True
    return sheet.loc[:, sheet_dtypes]

numeric_flowsheet = return_numeric(flowsheet)

flowsheet_pt_split = split_by_pt(flowsheet, 'project_id')



#### Think about how far apart the parameters are for each patient
def get_summary(parameter, sheet_pt, summary, arg = None):
    """ 
    Function to get some summary data \n
    Parameter is a column \n
    summary is a function to be run on that column
    """

    parameter = return_colname(parameter, sheet_pt[0])
    values = list()
    
    for i in range(len(sheet_pt)):
        values.append(summary(sheet_pt[i], parameter, arg))

    return values

def average_time(sheet, date_time, arg = None):
    """
    Function to return the average time between parameters
    """

    times = sheet[date_time]
    times = times.sort_values()
    times2 = pd.Series(pd.NaT)
    times2 = times2.append(times)
    times = times.append(pd.Series(pd.NaT))
    times2.index = times.index
    differences = times - times2
    return differences

def proportion_highfreq(sheet, date_time, arg):
    """
    Function to return the average time between parameters
    """

    times = sheet[date_time]
    times = times.sort_values()
    times2 = pd.Series(pd.NaT)
    times2 = times2.append(times)
    times = times.append(pd.Series(pd.NaT))
    times2.index = times.index
    differences = times - times2
    high_freq = sum(differences <= pd.Timedelta(arg))
    return high_freq/(sheet.shape[0] - 1)

less1h = get_summary('taken_datetime', flowsheet_pt_split, proportion_highfreq, '0 days 00:05:00')
sum(less1h)/len(less1h)


#Instantiate figure 
fig, axs = plt.subplots(5, 5, figsize=(15,15), sharex=True)

    #Work through the different patients
for i in range(5):
    for j in range(5):

        pt_sheet = flowsheet_pt_split[5*i + j + 25]
        differences = average_time(pt_sheet, 'taken_datetime')
        differences = [k.total_seconds() for k in differences]
        axs[i, j].plot(range(len(differences)), differences)

for ax in axs.flat:
    ax.set(xlabel='index', ylabel='time gap')

#Save it
fig.savefig('Project/PICU_project/figs/Time_gaps1.png')

def return_highfreq(sheet, date_time, arg):
    """ Function to return only the rows which average over a certain frequency
    """






#Do some plotting
def plot_PICU_params(df, patient, title="", xlabel='Date/Time', ylabel='Value', dpi=500):
    ''' 
    Function to do some plotting of timeseries data, plots multiple parameters for same patient
    Takes parameters df (pandas dataframe), x and y values, plots them
    You have to choose a label to print
    Uses the input from the index
    Should probably change this after making more composite values so only plotting composites
    '''

    #Choose columns which have fewest NaNs
    pt_df = df[patient]
    colnames = pt_df.columns
    numNaNs = np.zeros([len(colnames) - 3])

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = pt_df.iloc[:, (i)].isna() == False
        numNaNs[i - 3] = pt_df.loc[isnan, :].shape[0]

    #Sort the NaNs
    sortedNaNs = np.argsort(numNaNs)
    top25locs = sortedNaNs[range(-25, 0)]
    top25 = sortedNaNs[top25locs] + 3

    #Instantiate figure 
    fig, axs = plt.subplots(5, 5, figsize=(15,15), sharex=True)

    #Work through the different patients
    for i in range(5):
        for j in range(5):

            #Choose the correct column
            column = colnames[top25[5*i + j]]
            
            #First object in list for each pt is list of vars
            x = pt_df['taken_datetime']
            y = pt_df[column]
            axs[i, j].plot(x, y)
            axs[i, j].set_title(column)

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/PICU_plots' + str(patient) + '.png')

for i in range(15):
    plot_PICU_params(flowsheet_pt_split, i)